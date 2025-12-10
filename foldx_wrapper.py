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


class FoldXWrapper:
    """FoldX能量计算封装类"""
    
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
        """
        检查FoldX是否可用
        
        Returns:
            是否可用
        """
        try:
            result = subprocess.run(
                [self.foldx_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # FoldX 通常在没有参数时返回非0，但会有输出
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
        计算突变的ΔΔG（稳定性变化）
        
        Args:
            pdb_file: PDB文件路径
            mutations: 突变列表，格式 ['TA45C', 'SA125C']
            n_runs: FoldX运行次数（取平均）
            timeout: 超时时间（秒）
        
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
                    # FoldX格式：TA45C,SA125C;
                    mutation_str = ",".join(mutations)
                    f.write(f"{mutation_str};\n")
                
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
                    print(f"FoldX execution time: {elapsed:.1f}s")
                
                # 解析结果
                ddg = self._parse_foldx_output(temp_dir, Path(pdb_file).stem)
                
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
    
    def _parse_foldx_output(self, work_dir: str, pdb_stem: str) -> Optional[float]:
        """
        解析FoldX输出文件
        
        Args:
            work_dir: FoldX工作目录
            pdb_stem: PDB文件名（不含扩展名）
        
        Returns:
            ΔΔG值，失败返回None
        """
        # FoldX输出文件命名模式
        possible_files = [
            f"Average_{pdb_stem}_1.fxout",
            f"Average_{pdb_stem}.fxout",
            f"Dif_{pdb_stem}_1.fxout",
            f"Dif_{pdb_stem}.fxout"
        ]
        
        for filename in possible_files:
            filepath = os.path.join(work_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        lines = f.readlines()
                        
                        # 跳过头部，找到数据行
                        for line in lines:
                            if line.startswith('#') or not line.strip():
                                continue
                            
                            parts = line.strip().split()
                            
                            # FoldX输出格式：通常第二列是total energy
                            if len(parts) >= 2:
                                try:
                                    # 尝试解析能量值
                                    ddg = float(parts[1])
                                    return ddg
                                except ValueError:
                                    continue
                
                except Exception as e:
                    if self.verbose:
                        print(f"Error parsing {filename}: {str(e)}")
                    continue
        
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
    # 三字母到单字母映射
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
