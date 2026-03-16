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
                
                for rotabase_candidate in [
                    os.path.join(os.path.dirname(self.foldx_path), "rotabase.txt"),
                    "/usr/local/bin/rotabase.txt",
                    "/content/rotabase.txt",
                    "/content/foldx_extracted/rotabase.txt",
                ]:
                    if os.path.isfile(rotabase_candidate):
                        shutil.copy(rotabase_candidate, os.path.join(temp_dir, "rotabase.txt"))
                        if self.verbose:
                            print(f"  ✓ rotabase.txt 已复制到工作目录")
                        break
                else:
                    # 递归搜索
                    for search_dir in ["/content/foldx_extracted", "/content"]:
                        if os.path.isdir(search_dir):
                            for root, _dirs, _files in os.walk(search_dir):
                                if "rotabase.txt" in _files:
                                    shutil.copy(os.path.join(root, "rotabase.txt"),
                                                os.path.join(temp_dir, "rotabase.txt"))
                                    if self.verbose:
                                        print(f"  ✓ rotabase.txt 找到并复制: {root}")
                                    break
                            else:
                                continue
                            break
                    else:
                        if self.verbose:
                            print(f"  ⚠️ rotabase.txt 未找到! FoldX 将无法正常运行")
                
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
        all_files = os.listdir(work_dir)
        fxout_files = sorted([f for f in all_files if f.endswith('.fxout')])
        
        if self.verbose and fxout_files:
            print(f"  找到 {len(fxout_files)} 个 .fxout 文件: {fxout_files}")
        
        dif_files = [f for f in fxout_files if f.startswith('Dif_')]
        for dif_file in dif_files:
            ddg = self._parse_dif_file(os.path.join(work_dir, dif_file))
            if ddg is not None:
                return ddg
        
        avg_files = [f for f in fxout_files if f.startswith('Average_')]
        for avg_file in avg_files:
            ddg = self._parse_average_file(os.path.join(work_dir, avg_file))
            if ddg is not None:
                return ddg
        
        if self.verbose:
            print(f"  ⚠️ 未找到可解析的输出文件")
        
        return None
    
    def _parse_dif_file(self, filepath: str) -> Optional[float]:
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            if self.verbose:
                print(f"  解析Dif文件: {os.path.basename(filepath)} ({len(lines)}行)")
            
            for i, line in enumerate(lines):
                if line.startswith('#') or line.startswith('FoldX'):
                    continue
                
                if not line.strip():
                    continue
                
                # 跳过表头(包含"total energy"或"Pdb"字样)
                if 'total energy' in line or 'Pdb\t' in line:
                    if self.verbose:
                        print(f"    第{i+1}行: 表头,跳过")
                    continue
                
                parts = line.strip().split('\t')
                
                if len(parts) >= 2:
                    try:
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
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            if self.verbose:
                print(f"  解析Average文件: {os.path.basename(filepath)} ({len(lines)}行)")
            
            for i, line in enumerate(lines):
                if (line.startswith('#') or 
                    line.startswith('FoldX') or
                    not line.strip() or
                    'total energy' in line or
                    'Pdb\t' in line):
                    continue
                
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
    # 三字母到单字母映射
    AA_MAP = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    res1_letter = AA_MAP.get(res1_name.upper(), res1_name[0].upper())
    res2_letter = AA_MAP.get(res2_name.upper(), res2_name[0].upper())
    
    mutations = [
        f"{res1_letter}{chain}{res1_id}C",
        f"{res2_letter}{chain}{res2_id}C"
    ]
    
    return mutations


# ============================================================
# 测试和调试函数
# ============================================================

def test_foldx_installation(foldx_path: str = "/usr/local/bin/foldx"):
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
    
    test_foldx_installation()

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
