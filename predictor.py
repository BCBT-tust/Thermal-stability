"""
Disulfide Bond Predictor Module
二硫键预测主模块：整合所有功能，使用BioPython解析
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Polypeptide import is_aa

from .geometry import (
    check_disulfide_geometry,
    calculate_distance,
    calculate_bfactor_average,
    estimate_cb_position
)
from .filters import CompositeFilter
from .pdb_parser import three_to_one


class DisulfideCandidate:
    """二硫键候选数据类"""
    
    def __init__(self, chain: str, res1_id: int, res1_name: str,
                 res2_id: int, res2_name: str, geometry_data: dict):
        """
        Args:
            chain: 链ID
            res1_id, res2_id: 残基序列号
            res1_name, res2_name: 残基三字母名称
            geometry_data: 几何数据字典
        """
        self.chain = chain
        self.res1_id = res1_id
        self.res1_name = res1_name
        self.res2_id = res2_id
        self.res2_name = res2_name
        
        # 几何数据
        self.ca_distance = geometry_data.get('ca_distance')
        self.cb_distance = geometry_data.get('cb_distance')
        self.sg_distance = geometry_data.get('sg_distance')
        self.dihedral = geometry_data.get('dihedral')
        
        # 单字母代码
        self.res1_letter = three_to_one(res1_name)
        self.res2_letter = three_to_one(res2_name)
        
        # 突变描述
        self.mutation = f"{self.res1_letter}{res1_id}C+{self.res2_letter}{res2_id}C"
        
        # 序列距离
        self.seq_separation = abs(res2_id - res1_id)
        
        # 能量数据（后续填充）
        self.ddg = None
        self.foldx_success = False
        
        # B因子数据
        self.res1_bfactor = None
        self.res2_bfactor = None
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'chain': self.chain,
            'res1_id': self.res1_id,
            'res1_name': self.res1_name,
            'res1_letter': self.res1_letter,
            'res2_id': self.res2_id,
            'res2_name': self.res2_name,
            'res2_letter': self.res2_letter,
            'mutation': self.mutation,
            'seq_separation': self.seq_separation,
            'ca_distance': self.ca_distance,
            'cb_distance': self.cb_distance,
            'sg_distance': self.sg_distance,
            'dihedral': self.dihedral,
            'ddg': self.ddg,
            'res1_bfactor': self.res1_bfactor,
            'res2_bfactor': self.res2_bfactor
        }
    
    def __repr__(self):
        return (f"DisulfideCandidate({self.mutation}, "
                f"CB={self.cb_distance:.2f}Å, "
                f"ΔΔG={self.ddg if self.ddg else 'N/A'})")


class DisulfideBondPredictor:
    """
    二硫键预测器主类
    使用BioPython解析PDB并预测可行的二硫键位点
    """
    
    def __init__(self, pdb_file: str, 
                 catalytic_residues: Optional[List[Tuple[str, int]]] = None,
                 config: Optional[dict] = None):
        """
        初始化预测器
        
        Args:
            pdb_file: PDB文件路径
            catalytic_residues: 催化残基列表 [(chain, resid), ...]
            config: 配置字典（覆盖默认参数）
        """
        self.pdb_file = pdb_file
        self.catalytic_residues = catalytic_residues or []
        
        # 默认配置
        self.config = {
            'min_seq_separation': 3,
            'max_candidates': 20,
            'geometry_mode': 'relaxed',  # 'strict' or 'relaxed'
            'use_dbd2': False,
            'check_surface': False,
            'check_clash': False,
            'check_bfactor': False,
            'surface_threshold': 20.0,
            'clash_threshold': 5,
            'max_bfactor': 50.0
        }
        
        # 更新配置
        if config:
            self.config.update(config)
        
        # 加载结构
        self.structure = self._load_structure()
        
        # 初始化过滤器
        self.filters = CompositeFilter(
            min_seq_separation=self.config['min_seq_separation'],
            catalytic_residues=self.catalytic_residues,
            check_surface=self.config['check_surface'],
            check_clash=self.config['check_clash'],
            check_bfactor=self.config['check_bfactor'],
            surface_threshold=self.config['surface_threshold'],
            clash_threshold=self.config['clash_threshold'],
            max_bfactor=self.config['max_bfactor']
        )
        
        # 统计信息
        self.stats = {
            'total_pairs': 0,
            'filtered_by_seq': 0,
            'filtered_by_cys': 0,
            'filtered_by_catalytic': 0,
            'filtered_by_surface': 0,
            'filtered_by_geometry': 0,
            'filtered_by_clash': 0,
            'filtered_by_bfactor': 0,
            'final_candidates': 0
        }
    
    def _load_structure(self):
        """使用BioPython加载PDB结构"""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', self.pdb_file)
        return structure
    
    def _get_residue_atoms(self, residue) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        提取残基的CA和CB坐标
        
        Args:
            residue: BioPython的Residue对象
        
        Returns:
            (ca_coord, cb_coord)
        """
        ca_coord = None
        cb_coord = None
        
        try:
            if 'CA' in residue:
                ca_coord = residue['CA'].get_coord()
            
            if 'CB' in residue:
                cb_coord = residue['CB'].get_coord()
            elif residue.get_resname() != 'GLY':
                # 估算CB位置
                if 'CA' in residue and 'N' in residue and 'C' in residue:
                    ca = residue['CA'].get_coord()
                    n = residue['N'].get_coord()
                    c = residue['C'].get_coord()
                    cb_coord = estimate_cb_position(ca, n, c)
        except:
            pass
        
        return ca_coord, cb_coord
    
    def _get_residue_bfactor(self, residue) -> Optional[float]:
        """
        计算残基的平均B因子
        
        Args:
            residue: BioPython的Residue对象
        
        Returns:
            平均B因子
        """
        try:
            bfactors = [atom.get_bfactor() for atom in residue.get_atoms()]
            if bfactors:
                return float(np.mean(bfactors))
        except:
            pass
        return None
    
    def _collect_all_ca_coords(self, chain) -> List[np.ndarray]:
        """收集链上所有CA坐标（用于表面检查）"""
        ca_coords = []
        for residue in chain:
            if not is_aa(residue):
                continue
            if 'CA' in residue:
                ca_coords.append(residue['CA'].get_coord())
        return ca_coords
    
    def _collect_nearby_atoms(self, residue, radius: float = 5.0) -> List[np.ndarray]:
        """收集残基周围的所有原子坐标（用于碰撞检查）"""
        if 'CB' not in residue:
            return []
        
        cb_coord = residue['CB'].get_coord()
        nearby_atoms = []
        
        for model in self.structure:
            for chain in model:
                for res in chain:
                    if res == residue:
                        continue
                    for atom in res:
                        dist = calculate_distance(cb_coord, atom.get_coord())
                        if dist and dist < radius:
                            nearby_atoms.append(atom.get_coord())
        
        return nearby_atoms
    
    def predict(self) -> List[DisulfideCandidate]:
        """
        执行二硫键预测
        
        Returns:
            候选列表（已按CB距离排序）
        """
        candidates = []
        
        try:
            for model in self.structure:
                for chain in model:
                    # 收集氨基酸残基
                    residues = [res for res in chain if is_aa(res)]
                    
                    # 收集所有CA坐标（用于表面检查）
                    all_ca_coords = self._collect_all_ca_coords(chain) if self.config['check_surface'] else None
                    
                    # 遍历所有残基对
                    for i, res1 in enumerate(residues):
                        for j, res2 in enumerate(residues):
                            if j <= i:
                                continue
                            
                            self.stats['total_pairs'] += 1
                            
                            try:
                                # 基本信息
                                chain_id = chain.id
                                res1_id = res1.id[1]
                                res2_id = res2.id[1]
                                res1_name = res1.get_resname()
                                res2_name = res2.get_resname()
                                
                                # 提取坐标
                                ca1, cb1 = self._get_residue_atoms(res1)
                                ca2, cb2 = self._get_residue_atoms(res2)
                                
                                if ca1 is None or cb1 is None or ca2 is None or cb2 is None:
                                    continue
                                
                                # B因子
                                res1_bfactor = self._get_residue_bfactor(res1) if self.config['check_bfactor'] else None
                                res2_bfactor = self._get_residue_bfactor(res2) if self.config['check_bfactor'] else None
                                
                                # 收集周围原子（碰撞检查）
                                nearby1 = self._collect_nearby_atoms(res1) if self.config['check_clash'] else None
                                nearby2 = self._collect_nearby_atoms(res2) if self.config['check_clash'] else None
                                
                                # 应用过滤器
                                is_valid, reason = self.filters.apply_all(
                                    chain_id=chain_id,
                                    res1_id=res1_id,
                                    res2_id=res2_id,
                                    res1_name=res1_name,
                                    res2_name=res2_name,
                                    res1_ca=ca1,
                                    res1_cb=cb1,
                                    res2_ca=ca2,
                                    res2_cb=cb2,
                                    all_ca_coords=all_ca_coords,
                                    all_atoms_near_res1=nearby1,
                                    all_atoms_near_res2=nearby2,
                                    res1_bfactor=res1_bfactor,
                                    res2_bfactor=res2_bfactor
                                )
                                
                                # 记录过滤统计
                                if not is_valid:
                                    if reason == 'sequence_separation':
                                        self.stats['filtered_by_seq'] += 1
                                    elif reason == 'existing_cys_pair':
                                        self.stats['filtered_by_cys'] += 1
                                    elif reason == 'protected_residue':
                                        self.stats['filtered_by_catalytic'] += 1
                                    elif 'buried' in reason:
                                        self.stats['filtered_by_surface'] += 1
                                    elif 'clash' in reason:
                                        self.stats['filtered_by_clash'] += 1
                                    elif 'bfactor' in reason:
                                        self.stats['filtered_by_bfactor'] += 1
                                    continue
                                
                                # 几何可行性检查
                                is_geom_valid, geometry_data = check_disulfide_geometry(
                                    ca1, cb1, ca2, cb2,
                                    mode=self.config['geometry_mode'],
                                    use_dbd2=self.config['use_dbd2']
                                )
                                
                                if not is_geom_valid:
                                    self.stats['filtered_by_geometry'] += 1
                                    continue
                                
                                # 创建候选
                                candidate = DisulfideCandidate(
                                    chain=chain_id,
                                    res1_id=res1_id,
                                    res1_name=res1_name,
                                    res2_id=res2_id,
                                    res2_name=res2_name,
                                    geometry_data=geometry_data
                                )
                                
                                # 添加B因子数据
                                candidate.res1_bfactor = res1_bfactor
                                candidate.res2_bfactor = res2_bfactor
                                
                                candidates.append(candidate)
                            
                            except Exception as e:
                                # 跳过单个残基对的错误
                                continue
            
            # 按CB距离排序（越接近4.0Å越好）
            candidates.sort(key=lambda x: abs(x.cb_distance - 4.0))
            
            # 限制候选数量
            candidates = candidates[:self.config['max_candidates']]
            
            self.stats['final_candidates'] = len(candidates)
            
            return candidates
        
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_statistics(self) -> dict:
        """获取预测统计信息"""
        return self.stats.copy()
    
    def print_statistics(self):
        """打印统计信息"""
        print(f"\n{'='*60}")
        print(f"Prediction Statistics")
        print(f"{'='*60}")
        print(f"Total residue pairs: {self.stats['total_pairs']}")
        print(f"Filtered by sequence separation: {self.stats['filtered_by_seq']}")
        print(f"Filtered by existing Cys pairs: {self.stats['filtered_by_cys']}")
        print(f"Filtered by catalytic protection: {self.stats['filtered_by_catalytic']}")
        
        if self.config['check_surface']:
            print(f"Filtered by surface check: {self.stats['filtered_by_surface']}")
        
        print(f"Filtered by geometry: {self.stats['filtered_by_geometry']}")
        
        if self.config['check_clash']:
            print(f"Filtered by clash detection: {self.stats['filtered_by_clash']}")
        
        if self.config['check_bfactor']:
            print(f"Filtered by B-factor: {self.stats['filtered_by_bfactor']}")
        
        print(f"Final candidates: {self.stats['final_candidates']}")
        print(f"{'='*60}\n")


def generate_report(candidates: List[DisulfideCandidate], 
                   output_file: str,
                   config: dict = None):
    """
    生成预测报告
    
    Args:
        candidates: 候选列表
        output_file: 输出文件路径
        config: 配置信息
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Disulfide Bond Prediction Report\n")
        f.write("二硫键预测报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total candidates: {len(candidates)}\n")
        
        if config:
            f.write(f"\nConfiguration:\n")
            f.write(f"  Geometry mode: {config.get('geometry_mode', 'N/A')}\n")
            f.write(f"  DbD2 logic: {'Enabled' if config.get('use_dbd2') else 'Disabled'}\n")
            f.write(f"  Surface check: {'Enabled' if config.get('check_surface') else 'Disabled'}\n")
            f.write(f"  Clash check: {'Enabled' if config.get('check_clash') else 'Disabled'}\n")
        
        f.write("=" * 80 + "\n\n")
        
        if len(candidates) == 0:
            f.write("❌ No candidates found.\n")
            f.write("\nSuggestions:\n")
            f.write("- Try relaxed geometry mode (CB: 3.0-5.0Å)\n")
            f.write("- Disable surface check\n")
            f.write("- Disable clash check\n")
            f.write("- Reduce min_seq_separation\n")
        else:
            for i, cand in enumerate(candidates, 1):
                f.write(f"Candidate {i}:\n")
                f.write(f"  Mutation: {cand.mutation}\n")
                f.write(f"  Position: {cand.res1_name}{cand.res1_id} - {cand.res2_name}{cand.res2_id}\n")
                f.write(f"  Chain: {cand.chain}\n")
                f.write(f"  Sequence separation: {cand.seq_separation}\n")
                f.write(f"  CA-CA distance: {cand.ca_distance:.2f} Å\n")
                f.write(f"  CB-CB distance: {cand.cb_distance:.2f} Å (ideal: 4.0 Å)\n")
                
                if cand.sg_distance:
                    f.write(f"  SG-SG distance: {cand.sg_distance:.2f} Å (DbD2)\n")
                
                if cand.dihedral is not None:
                    f.write(f"  CA-CB-CB-CA dihedral: {cand.dihedral:.1f}°\n")
                
                if cand.res1_bfactor is not None:
                    f.write(f"  B-factors: {cand.res1_bfactor:.1f} / {cand.res2_bfactor:.1f}\n")
                
                if cand.ddg is not None:
                    f.write(f"  ΔΔG: {cand.ddg:.2f} kcal/mol (FoldX)\n")
                
                f.write("-" * 80 + "\n")
            
            # 推荐优先级
            f.write("\nRecommended Priority (sorted by CB distance to 4.0Å):\n")
            for i, cand in enumerate(candidates[:5], 1):
                deviation = abs(cand.cb_distance - 4.0)
                f.write(f"  {i}. {cand.mutation} (deviation: {deviation:.2f}Å)\n")
    
    print(f"Report saved to: {output_file}")
