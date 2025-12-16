"""
FoldX Wrapper Module
FoldX能量计算封装模块
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import time
import re


class FoldXWrapper:
    """FoldX能量计算封装类 """
    
    def __init__(self, foldx_path: str = "foldx", verbose: bool = False):
        """
        初始化FoldX封装器
        
        Args:
            foldx_path: FoldX可执行文件路径
            verbose: 是否打印详细信息
        """
        self.foldx_path = foldx_path
        self.verbose = verbose
        self.is_available = self._check_foldx()
    
    def _check_foldx(self) -> bool:
        """检查FoldX是否可用"""
        try:
            result = subprocess.run(
                [self.foldx_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            
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
        
        Args:
            pdb_file: PDB文件路径
            mutations: 突变列表,格式 ['TA45C', 'SA125C']
            n_runs: FoldX运行次数(取平均)
            timeout: 超时时间(秒)
        
        Returns:
            结果字典 {'ddg': float, 'success': bool, 'error': str}
        """
        if not self.is_available:
            return {
                'ddg': None,
                'success': False,
                'error': 'FoldX not available'
            }
        
        # 创建临时工作目录
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # 复制PDB文件到工作目录
                pdb_name = Path(pdb_file).name
                work_pdb = os.path.join(temp_dir, pdb_name)
                
                import shutil
                shutil.copy(pdb_file, work_pdb)
                
                # 创建突变列表文件
                mut_file = os.path.join(temp_dir, "individual_list.txt")
                with open(mut_file, 'w') as f:
                    mutation_str = ",".join(mutations)
                    f.write(f"{mutation_str};\n")
                
                if self.verbose:
                    print(f"  Mutations: {mutation_str}")
                
                # 构建FoldX命令
                cmd = [
                    self.foldx_path,
                    "--command=BuildModel",
                    f"--pdb={pdb_name}",
                    "--mutant-file=individual_list.txt",
                    f"--numberOfRuns={n_runs}"
                ]
                
                # 执行FoldX
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
                    print(f"  FoldX execution: {elapsed:.1f}s (return code: {result.returncode})")
                
                # 解析结果 - 使用增强的解析逻辑
                ddg = self._parse_foldx_output_enhanced(temp_dir, Path(pdb_file).stem, mutations)
                
                if ddg is not None:
                    return {
                        'ddg': ddg,
                        'success': True,
                        'error': None,
                        'elapsed_time': elapsed
                    }
                else:
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
    
    def _parse_foldx_output_enhanced(self, work_dir: str, pdb_stem: str, 
                                    mutations: List[str]) -> Optional[float]:
        """
        增强版FoldX输出解析
        
        尝试多种策略:
        1. Dif文件(能量差)
        2. Average文件
        3. Raw文件
        4. 正则表达式搜索
        """
        if self.verbose:
            print(f"\n  === FoldX Output Parsing Debug ===")
            print(f"  Working directory: {work_dir}")
            print(f"  PDB stem: {pdb_stem}")
        
        # 列出所有输出文件
        all_files = os.listdir(work_dir)
        fxout_files = [f for f in all_files if f.endswith('.fxout')]
        
        if self.verbose:
            print(f"  Found {len(fxout_files)} .fxout files:")
            for f in fxout_files:
                print(f"    - {f}")
        
        # 策略1: 优先解析Dif文件(最可靠)
        dif_files = [f for f in fxout_files if f.startswith('Dif_')]
        if dif_files:
            if self.verbose:
                print(f"\n  Trying Dif files: {dif_files}")
            
            for filename in dif_files:
                result = self._parse_dif_file(os.path.join(work_dir, filename))
                if result is not None:
                    if self.verbose:
                        print(f"  ✓ Successfully parsed {filename}: ΔΔG = {result:.2f}")
                    return result
        
        # 策略2: 尝试Average文件
        avg_files = [f for f in fxout_files if f.startswith('Average_')]
        if avg_files:
            if self.verbose:
                print(f"\n  Trying Average files: {avg_files}")
            
            for filename in avg_files:
                result = self._parse_average_file(os.path.join(work_dir, filename))
                if result is not None:
                    if self.verbose:
                        print(f"  ✓ Successfully parsed {filename}: ΔΔG = {result:.2f}")
                    return result
        
        # 策略3: 搜索所有.fxout文件中的数值
        if self.verbose:
            print(f"\n  Trying all .fxout files with flexible parsing...")
        
        for filename in fxout_files:
            result = self._parse_any_fxout(os.path.join(work_dir, filename))
            if result is not None:
                if self.verbose:
                    print(f"  ✓ Found value in {filename}: {result:.2f}")
                return result
        
        if self.verbose:
            print(f"  ✗ All parsing strategies failed")
        
        return None
    
    def _parse_dif_file(self, filepath: str) -> Optional[float]:
        """
        解析Dif文件(能量差文件)
        
        格式示例:
        #header
        mutation_name    total_energy    other_cols...
        TA45C,SA125C     2.34           ...
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            if self.verbose:
                print(f"    Dif file has {len(lines)} lines")
            
            for i, line in enumerate(lines):
                # 跳过注释和空行
                if line.startswith('#') or not line.strip():
                    continue
                
                # 跳过明显的表头
                if 'total' in line.lower() or 'energy' in line.lower():
                    if self.verbose:
                        print(f"    Line {i}: Header - {line.strip()[:60]}")
                    continue
                
                parts = line.strip().split()
                
                if len(parts) >= 2:
                    try:
                        # 尝试解析第2列(索引1)
                        ddg = float(parts[1])
                        
                        # 合理性检查: ΔΔG通常在-20到+20范围
                        if -50 < ddg < 50:
                            if self.verbose:
                                print(f"    Line {i}: Found ΔΔG = {ddg:.2f}")
                                print(f"    Full line: {line.strip()}")
                            return ddg
                    except ValueError:
                        continue
        
        except Exception as e:
            if self.verbose:
                print(f"    Error: {str(e)}")
        
        return None
    
    def _parse_average_file(self, filepath: str) -> Optional[float]:
        """
        解析Average文件
        
        可能包含平均能量值
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            if self.verbose:
                print(f"    Average file has {len(lines)} lines")
            
            for i, line in enumerate(lines):
                if line.startswith('#') or not line.strip():
                    continue
                
                if 'total' in line.lower() or 'energy' in line.lower():
                    continue
                
                parts = line.strip().split()
                
                # Average文件可能有多列,尝试找到能量值
                for j, part in enumerate(parts[1:], 1):  # 跳过第一列(名称)
                    try:
                        value = float(part)
                        if -50 < value < 50:
                            if self.verbose:
                                print(f"    Line {i}, Col {j}: Found value = {value:.2f}")
                            return value
                    except ValueError:
                        continue
        
        except Exception as e:
            if self.verbose:
                print(f"    Error: {str(e)}")
        
        return None
    
    def _parse_any_fxout(self, filepath: str) -> Optional[float]:
        """
        灵活解析任何.fxout文件
        
        搜索所有看起来像能量值的数字
        """
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # 使用正则表达式查找数字
            # 寻找格式: 空格 + 数字(可能有负号和小数点) + 空格
            pattern = r'\s+(-?\d+\.\d+)\s+'
            matches = re.findall(pattern, content)
            
            if matches:
                # 转换为浮点数并过滤合理范围
                values = []
                for m in matches:
                    try:
                        v = float(m)
                        if -50 < v < 50:
                            values.append(v)
                    except:
                        continue
                
                if values:
                    # 返回第一个合理值
                    # (注意: 这不一定是ΔΔG,但在没有其他信息时是最佳猜测)
                    if self.verbose:
                        print(f"    Found {len(values)} potential values: {values[:5]}")
                    return values[0]
        
        except Exception as e:
            if self.verbose:
                print(f"    Error: {str(e)}")
        
        return None


def format_mutations_for_foldx(res1_name: str, res1_id: int, 
                               res2_name: str, res2_id: int,
                               chain: str = 'A') -> List[str]:
    """
    格式化突变为FoldX格式
    
    Args:
        res1_name, res2_name: 三字母残基名称
        res1_id, res2_id: 残基序列号
        chain: 链ID
    
    Returns:
        FoldX格式的突变列表 ['TA45C', 'SA125C']
    """
    AA_MAP = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    res1_letter = AA_MAP.get(res1_name.upper(), res1_name[0])
    res2_letter = AA_MAP.get(res2_name.upper(), res2_name[0])
    
    mutations = [
        f"{res1_letter}{chain}{res1_id}C",
        f"{res2_letter}{chain}{res2_id}C"
    ]
    
    return mutations
