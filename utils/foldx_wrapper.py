"""
FoldX Wrapper Module
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import time


class FoldXWrapper:
    
    def __init__(self, foldx_path: str = "foldx", verbose: bool = False):

        self.foldx_path = foldx_path
        self.verbose = verbose
        self.is_available = self._check_foldx()
    
    def _check_foldx(self) -> bool:

        try:
            result = subprocess.run(
                [self.foldx_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # FoldX通常在没有参数时返回非0,但会有输出
            if result.stdout or result.stderr:
                if self.verbose:
                    print(f"✓ FoldX available at: {self.foldx_path}")
                return True
            else:
                if self.verbose:
                    print(f"✗ FoldX not responding: {self.foldx_path}")
                return False
        except FileNotFoundError:
            if self.verbose:
                print(f"✗ FoldX not found: {self.foldx_path}")
            return False
        except Exception as e:
            if self.verbose:
                print(f"✗ FoldX check failed: {str(e)}")
            return False
    
    def calculate_ddg(self, pdb_file: str, mutations: List[str], 
                     n_runs: int = 3, timeout: int = 300) -> Dict:
        """
        计算突变的ΔΔG(稳定性变化)
        """
        if not self.is_available:
            return {
                'ddg': None,
                'success': False,
                'error': 'FoldX not available'
            }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                pdb_name = Path(pdb_file).name
                work_pdb = os.path.join(temp_dir, pdb_name)
                
                import shutil
                shutil.copy(pdb_file, work_pdb)
                
                mut_file = os.path.join(temp_dir, "individual_list.txt")
                with open(mut_file, 'w') as f:
                    # FoldX格式: SA49C,SA68C;
                    mutation_str = ",".join(mutations)
                    f.write(f"{mutation_str};\n")
                
                if self.verbose:
                    print(f"  Mutant: {mutation_str}")

                cmd = [
                    self.foldx_path,
                    "--command=BuildModel",
                    f"--pdb={pdb_name}",
                    "--mutant-file=individual_list.txt",
                    f"--numberOfRuns={n_runs}"
                ]
                
                start_time = time.time()
                
                result = subprocess.run(
                    cmd,
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                elapsed = time.time() - start_time
                
                if self.verbose:
                    print(f"  FoldX执行时间: {elapsed:.1f}秒")
                
                ddg = self._parse_foldx_output(temp_dir, Path(pdb_file).stem)
                
                if ddg is not None:
                    return {
                        'ddg': ddg,
                        'success': True,
                        'error': None,
                        'elapsed_time': elapsed
                    }
                else:
                    # 调试信息
                    if self.verbose:
                        print(f"  ⚠️ 解析失败,检查输出文件:")
                        for f in os.listdir(temp_dir):
                            if f.endswith('.fxout'):
                                print(f"    - {f}")
                    
                    return {
                        'ddg': None,
                        'success': False,
                        'error': 'Failed to parse FoldX output',
                        'elapsed_time': elapsed
                    }
            
            except subprocess.TimeoutExpired:
                return {
                    'ddg': None,
                    'success': False,
                    'error': f'Timeout after {timeout}s'
                }
            
            except Exception as e:
                return {
                    'ddg': None,
                    'success': False,
                    'error': str(e)
                }
    
    def _parse_foldx_output(self, work_dir: str, pdb_stem: str) -> Optional[float]:
        """
        解析FoldX输出文件 - 修复版
        
        优先查找Dif文件(能量差文件),正确解析tab分隔的数据
        
        Args:
            work_dir: FoldX工作目录
            pdb_stem: PDB文件名(不含扩展名)
        
        Returns:
            ΔΔG值,失败返回None
        """
        # 优先级1: Dif文件(最可靠)
        dif_file = os.path.join(work_dir, f"Dif_{pdb_stem}.fxout")
        
        if os.path.exists(dif_file):
            ddg = self._parse_dif_file(dif_file)
            if ddg is not None:
                return ddg
        
        # 优先级2: Average文件
        avg_file = os.path.join(work_dir, f"Average_{pdb_stem}.fxout")
        
        if os.path.exists(avg_file):
            ddg = self._parse_average_file(avg_file)
            if ddg is not None:
                return ddg
        
        # 都失败了
        if self.verbose:
            print(f"  ⚠️ 未找到可解析的输出文件")
        
        return None
    
    def _parse_dif_file(self, filepath: str) -> Optional[float]:
        """
        解析Dif文件(能量差文件)
        
        Dif文件格式(tab分隔):
        Line 1-8: 头部信息
        Line 9: 表头(Pdb  total energy  Backbone Hbond  ...)
        Line 10: 数据(XynA_1.pdb  0.219481  -0.44596  ...)
                              ^^^^^^^^ 这是ΔΔG
        
        Args:
            filepath: Dif文件路径
        
        Returns:
            ΔΔG值
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            if self.verbose:
                print(f"  解析Dif文件: {os.path.basename(filepath)} ({len(lines)}行)")
            
            for i, line in enumerate(lines):
                # 跳过注释
                if line.startswith('#') or line.startswith('FoldX'):
                    continue
                
                # 跳过空行
                if not line.strip():
                    continue
                
                # 跳过表头(包含"total energy"或"Pdb"字样)
                if 'total energy' in line or 'Pdb\t' in line:
                    if self.verbose:
                        print(f"    第{i+1}行: 表头,跳过")
                    continue
                
                # 解析数据行
                # 注意:FoldX使用tab分隔,不是空格!
                parts = line.strip().split('\t')
                
                if len(parts) >= 2:
                    try:
                        # 第1列: PDB文件名
                        # 第2列: total energy差值(ΔΔG)
                        pdb_name = parts[0]
                        ddg = float(parts[1])
                        
                        if self.verbose:
                            print(f"    第{i+1}行: {pdb_name} → ΔΔG = {ddg:.3f} kcal/mol ✓")
                        
                        return ddg
                    
                    except (ValueError, IndexError) as e:
                        if self.verbose:
                            print(f"    第{i+1}行: 解析失败 ({str(e)})")
                        continue
        
        except Exception as e:
            if self.verbose:
                print(f"  ✗ 读取Dif文件失败: {str(e)}")
        
        return None
    
    def _parse_average_file(self, filepath: str) -> Optional[float]:
        """
        解析Average文件
        
        格式类似Dif文件,但包含标准差列
        
        Args:
            filepath: Average文件路径
        
        Returns:
            ΔΔG平均值
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            if self.verbose:
                print(f"  解析Average文件: {os.path.basename(filepath)} ({len(lines)}行)")
            
            for i, line in enumerate(lines):
                # 同样的过滤逻辑
                if (line.startswith('#') or 
                    line.startswith('FoldX') or
                    not line.strip() or
                    'total energy' in line or
                    'Pdb\t' in line):
                    continue
                
                # Tab分隔
                parts = line.strip().split('\t')
                
                if len(parts) >= 3:
                    try:
                        # Average文件格式:
                        # 第1列: PDB名
                        # 第2列: 标准差(SD)
                        # 第3列: total energy
                        pdb_name = parts[0]
                        sd = parts[1]
                        ddg = float(parts[2])
                        
                        if self.verbose:
                            print(f"    第{i+1}行: {pdb_name} → ΔΔG = {ddg:.3f} (SD={sd}) ✓")
                        
                        return ddg
                    
                    except (ValueError, IndexError) as e:
                        if self.verbose:
                            print(f"    第{i+1}行: 解析失败 ({str(e)})")
                        continue
        
        except Exception as e:
            if self.verbose:
                print(f"  ✗ 读取Average文件失败: {str(e)}")
        
        return None


def format_mutations_for_foldx(res1_name: str, res1_id: int, 
                               res2_name: str, res2_id: int,
                               chain: str = 'A') -> List[str]:
    """
    格式化突变为FoldX格式
    
    Args:
        res1_name: 残基1的三字母名称(如'SER')
        res1_id: 残基1的序列号(如49)
        res2_name: 残基2的三字母名称(如'SER')
        res2_id: 残基2的序列号(如68)
        chain: 链ID(默认'A')
    
    Returns:
        FoldX格式的突变列表
        例如: ['SA49C', 'SA68C']
        格式说明: {原残基单字母}{链ID}{位置}{目标残基}
    
    Examples:
        >>> format_mutations_for_foldx('SER', 49, 'SER', 68, 'A')
        ['SA49C', 'SA68C']
        
        >>> format_mutations_for_foldx('ALA', 244, 'SER', 284, 'A')
        ['AA244C', 'SA284C']
    """
    # 三字母到单字母映射
    AA_MAP = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    # 转换为单字母代码
    res1_letter = AA_MAP.get(res1_name.upper(), res1_name[0].upper())
    res2_letter = AA_MAP.get(res2_name.upper(), res2_name[0].upper())
    
    # FoldX格式: {原残基}{链ID}{位置}{目标残基}
    # 例如: SA49C 表示 Ser at A49 -> Cys
    mutations = [
        f"{res1_letter}{chain}{res1_id}C",
        f"{res2_letter}{chain}{res2_id}C"
    ]
    
    return mutations


# ============================================================
# 测试和调试函数
# ============================================================

def test_foldx_installation(foldx_path: str = "/usr/local/bin/foldx"):
    """
    测试FoldX是否正确安装
    
    Args:
        foldx_path: FoldX可执行文件路径
    
    Returns:
        测试结果字典
    """
    print("="*70)
    print("FoldX 安装测试")
    print("="*70)
    
    wrapper = FoldXWrapper(foldx_path=foldx_path, verbose=True)
    
    result = {
        'installed': wrapper.is_available,
        'path': foldx_path
    }
    
    if wrapper.is_available:
        print("\n✅ FoldX 安装正常")
    else:
        print("\n❌ FoldX 未正确安装")
        print("\n建议:")
        print("1. 检查文件是否存在")
        print("2. 检查文件权限(chmod +x)")
        print("3. 尝试手动运行: /usr/local/bin/foldx")
    
    print("="*70)
    
    return result


def debug_foldx_output(work_dir: str, pdb_stem: str):
    """
    调试FoldX输出文件
    
    显示所有.fxout文件的内容,帮助理解FoldX输出格式
    
    Args:
        work_dir: FoldX工作目录
        pdb_stem: PDB文件名(不含扩展名)
    """
    print("="*70)
    print(f"FoldX 输出调试: {pdb_stem}")
    print("="*70)
    
    # 查找所有.fxout文件
    fxout_files = [f for f in os.listdir(work_dir) if f.endswith('.fxout')]
    
    if not fxout_files:
        print("❌ 未找到.fxout文件")
        return
    
    print(f"\n找到 {len(fxout_files)} 个输出文件:\n")
    
    for filename in sorted(fxout_files):
        filepath = os.path.join(work_dir, filename)
        
        print(f"{'='*70}")
        print(f"📄 {filename}")
        print(f"{'='*70}")
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        print(f"总行数: {len(lines)}\n")
        
        # 显示所有内容
        for i, line in enumerate(lines, 1):
            print(f"{i:3d} | {line.rstrip()}")
        
        print()


if __name__ == "__main__":
    print("FoldX Wrapper Module - Test Model")
    print()
    
    # 测试安装
    test_foldx_installation()
    
    # 测试突变格式化
    print("\n突变格式化测试:")
    print("-"*70)
    
    test_cases = [
        ('SER', 49, 'SER', 68, 'A'),
        ('ALA', 244, 'SER', 284, 'A'),
        ('ASP', 102, 'THR', 105, 'A'),
    ]
    
    for res1, id1, res2, id2, chain in test_cases:
        mutations = format_mutations_for_foldx(res1, id1, res2, id2, chain)
        print(f"{res1}{id1} + {res2}{id2} → {mutations}")
