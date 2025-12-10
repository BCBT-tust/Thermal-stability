"""
Filters Module
筛选器模块：序列距离、催化残基保护、表面残基、空间碰撞
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from geometry import calculate_distance


class SequenceFilter:
    """序列距离过滤器"""
    
    def __init__(self, min_separation: int = 3):
        """
        Args:
            min_separation: 最小序列距离
        """
        self.min_separation = min_separation
    
    def is_valid(self, res1_id: int, res2_id: int) -> bool:
        """
        检查两个残基的序列距离是否满足要求
        
        Args:
            res1_id: 残基1的序列号
            res2_id: 残基2的序列号
        
        Returns:
            是否通过筛选
        """
        return abs(res1_id - res2_id) >= self.min_separation


class CatalyticFilter:
    """催化残基保护过滤器"""
    
    def __init__(self, catalytic_residues: List[Tuple[str, int]], 
                 buffer: int = 2):
        """
        Args:
            catalytic_residues: 催化残基列表 [(chain, resid), ...]
            buffer: 保护范围（±buffer个残基）
        """
        self.catalytic_residues = catalytic_residues or []
        self.buffer = buffer
    
    def is_protected(self, chain_id: str, res_id: int) -> Tuple[bool, Optional[str]]:
        """
        检查残基是否受保护
        
        Args:
            chain_id: 链ID
            res_id: 残基序列号
        
        Returns:
            (is_protected, reason)
        """
        res_tuple = (chain_id, res_id)
        
        # 直接是催化残基
        if res_tuple in self.catalytic_residues:
            return True, "catalytic"
        
        # 在催化残基附近
        for cat_chain, cat_res in self.catalytic_residues:
            if chain_id == cat_chain:
                if abs(res_id - cat_res) <= self.buffer:
                    return True, "near_catalytic"
        
        return False, None
    
    def is_valid_pair(self, chain_id: str, res1_id: int, res2_id: int) -> bool:
        """
        检查残基对是否可用（两个都不受保护）
        
        Args:
            chain_id: 链ID
            res1_id, res2_id: 两个残基的序列号
        
        Returns:
            是否通过筛选
        """
        protected1, _ = self.is_protected(chain_id, res1_id)
        protected2, _ = self.is_protected(chain_id, res2_id)
        
        return not (protected1 or protected2)


class SurfaceFilter:
    """表面残基过滤器（基于周围原子密度）"""
    
    def __init__(self, threshold: float = 20.0, radius: float = 8.0):
        """
        Args:
            threshold: 周围原子数阈值（小于此值认为是表面残基）
            radius: 检测半径（Å）
        """
        self.threshold = threshold
        self.radius = radius
    
    def is_surface(self, target_ca: np.ndarray, 
                   all_ca_coords: List[np.ndarray]) -> bool:
        """
        判断残基是否在表面
        
        Args:
            target_ca: 目标残基的CA坐标
            all_ca_coords: 所有残基的CA坐标列表
        
        Returns:
            是否为表面残基
        """
        if target_ca is None:
            return False
        
        nearby_count = 0
        
        for ca_coord in all_ca_coords:
            if ca_coord is None:
                continue
            
            dist = calculate_distance(target_ca, ca_coord)
            
            # 排除自身（距离接近0）
            if dist and 0.1 < dist < self.radius:
                nearby_count += 1
        
        # 周围原子少 → 表面残基
        return nearby_count < self.threshold


class ClashFilter:
    """空间碰撞过滤器"""
    
    def __init__(self, clash_threshold: int = 5, clash_radius: float = 3.0):
        """
        Args:
            clash_threshold: 碰撞原子数阈值（超过认为有碰撞）
            clash_radius: 碰撞检测半径（Å）
        """
        self.clash_threshold = clash_threshold
        self.clash_radius = clash_radius
    
    def has_clash(self, target_cb: np.ndarray, 
                  nearby_atoms: List[np.ndarray]) -> bool:
        """
        检测CB位置是否有空间碰撞
        
        Args:
            target_cb: 目标残基的CB坐标
            nearby_atoms: 附近所有原子的坐标列表
        
        Returns:
            是否存在碰撞
        """
        if target_cb is None:
            return False
        
        clash_count = 0
        
        for atom_coord in nearby_atoms:
            if atom_coord is None:
                continue
            
            dist = calculate_distance(target_cb, atom_coord)
            
            # 过近的原子
            if dist and dist < self.clash_radius:
                clash_count += 1
        
        return clash_count > self.clash_threshold


class CysteineFilter:
    """Cys残基过滤器（排除已有的二硫键）"""
    
    @staticmethod
    def is_existing_cys_pair(res1_name: str, res2_name: str) -> bool:
        """
        检查是否为已存在的Cys-Cys对
        
        Args:
            res1_name, res2_name: 两个残基的三字母代码
        
        Returns:
            是否为已存在的Cys对
        """
        return res1_name == 'CYS' and res2_name == 'CYS'


class BFactorFilter:
    """B因子过滤器（排除高度柔性区域）"""
    
    def __init__(self, max_bfactor: float = 50.0):
        """
        Args:
            max_bfactor: 最大允许的B因子值
        """
        self.max_bfactor = max_bfactor
    
    def is_valid(self, bfactor: Optional[float]) -> bool:
        """
        检查B因子是否在合理范围
        
        Args:
            bfactor: 残基的平均B因子
        
        Returns:
            是否通过筛选
        """
        if bfactor is None:
            return True  # 无B因子信息时默认通过
        
        return bfactor <= self.max_bfactor


class CompositeFilter:
    """
    组合过滤器：整合所有筛选条件
    """
    
    def __init__(self, 
                 min_seq_separation: int = 3,
                 catalytic_residues: List[Tuple[str, int]] = None,
                 check_surface: bool = True,
                 check_clash: bool = True,
                 check_bfactor: bool = False,
                 surface_threshold: float = 20.0,
                 clash_threshold: int = 5,
                 max_bfactor: float = 50.0):
        """
        初始化组合过滤器
        
        Args:
            min_seq_separation: 最小序列距离
            catalytic_residues: 催化残基列表
            check_surface: 是否检查表面残基
            check_clash: 是否检查空间碰撞
            check_bfactor: 是否检查B因子
            surface_threshold: 表面残基判断阈值
            clash_threshold: 碰撞判断阈值
            max_bfactor: 最大B因子
        """
        self.seq_filter = SequenceFilter(min_seq_separation)
        self.cat_filter = CatalyticFilter(catalytic_residues or [])
        self.surface_filter = SurfaceFilter(surface_threshold) if check_surface else None
        self.clash_filter = ClashFilter(clash_threshold) if check_clash else None
        self.bfactor_filter = BFactorFilter(max_bfactor) if check_bfactor else None
        
        self.check_surface = check_surface
        self.check_clash = check_clash
        self.check_bfactor = check_bfactor
    
    def apply_all(self, 
                  chain_id: str,
                  res1_id: int, res2_id: int,
                  res1_name: str, res2_name: str,
                  res1_ca: np.ndarray, res1_cb: np.ndarray,
                  res2_ca: np.ndarray, res2_cb: np.ndarray,
                  all_ca_coords: List[np.ndarray] = None,
                  all_atoms_near_res1: List[np.ndarray] = None,
                  all_atoms_near_res2: List[np.ndarray] = None,
                  res1_bfactor: Optional[float] = None,
                  res2_bfactor: Optional[float] = None) -> Tuple[bool, str]:
        """
        应用所有过滤器
        
        Returns:
            (is_valid, reject_reason)
        """
        # 1. 序列距离
        if not self.seq_filter.is_valid(res1_id, res2_id):
            return False, "sequence_separation"
        
        # 2. Cys对
        if CysteineFilter.is_existing_cys_pair(res1_name, res2_name):
            return False, "existing_cys_pair"
        
        # 3. 催化残基保护
        if not self.cat_filter.is_valid_pair(chain_id, res1_id, res2_id):
            return False, "protected_residue"
        
        # 4. 表面残基（可选）
        if self.check_surface and self.surface_filter and all_ca_coords:
            if not self.surface_filter.is_surface(res1_ca, all_ca_coords):
                return False, "buried_residue1"
            if not self.surface_filter.is_surface(res2_ca, all_ca_coords):
                return False, "buried_residue2"
        
        # 5. 空间碰撞（可选）
        if self.check_clash and self.clash_filter:
            if all_atoms_near_res1 and self.clash_filter.has_clash(res1_cb, all_atoms_near_res1):
                return False, "clash_residue1"
            if all_atoms_near_res2 and self.clash_filter.has_clash(res2_cb, all_atoms_near_res2):
                return False, "clash_residue2"
        
        # 6. B因子（可选）
        if self.check_bfactor and self.bfactor_filter:
            if not self.bfactor_filter.is_valid(res1_bfactor):
                return False, "high_bfactor_res1"
            if not self.bfactor_filter.is_valid(res2_bfactor):
                return False, "high_bfactor_res2"
        
        return True, "passed"
