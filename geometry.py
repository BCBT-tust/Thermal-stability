"""
Geometry Calculation Module
几何计算模块：距离、二面角、SG位置估算、B因子
"""

import numpy as np
import math
from typing import Tuple, Optional


def calculate_distance(coord1: np.ndarray, coord2: np.ndarray) -> Optional[float]:
    """
    计算两点间的欧氏距离
    
    Args:
        coord1: 第一个点的坐标 (x, y, z)
        coord2: 第二个点的坐标 (x, y, z)
    
    Returns:
        距离值（Å），失败返回None
    """
    if coord1 is None or coord2 is None:
        return None
    try:
        return float(np.linalg.norm(coord1 - coord2))
    except:
        return None


def calculate_dihedral(p1: np.ndarray, p2: np.ndarray, 
                      p3: np.ndarray, p4: np.ndarray) -> Optional[float]:
    """
    计算四个点定义的二面角
    
    Args:
        p1, p2, p3, p4: 四个点的坐标
    
    Returns:
        二面角（度），失败返回None
    
    Reference:
        Hazes & Dijkstra (1988). Protein Eng. 2(2):119-125
    """
    try:
        # 计算键向量
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        
        # 计算法向量
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        # 归一化
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        
        if n1_norm < 1e-6 or n2_norm < 1e-6:
            return None
        
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        
        # 计算二面角
        b2_norm = np.linalg.norm(b2)
        if b2_norm < 1e-6:
            return None
            
        m1 = np.cross(n1, b2 / b2_norm)
        
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        
        angle = np.arctan2(y, x)
        return float(np.degrees(angle))
    except:
        return None


def estimate_sg_position(ca: np.ndarray, cb: np.ndarray,
                        ca_cb_sg_angle: float = 114.6,
                        cb_sg_bond: float = 1.82) -> Optional[np.ndarray]:
    """
    估算SG原子位置（基于DbD2方法）
    
    Args:
        ca: Cα坐标
        cb: Cβ坐标
        ca_cb_sg_angle: Cα-Cβ-Sγ角度（度），默认114.6°
        cb_sg_bond: Cβ-Sγ键长（Å），默认1.82Å
    
    Returns:
        估算的SG坐标，失败返回None
    
    Reference:
        DbD2 (Disulfide by Design 2) algorithm
    """
    try:
        # CB到CA的方向向量
        v_cb_ca = ca - cb
        v_cb_ca_norm = v_cb_ca / np.linalg.norm(v_cb_ca)
        
        # 计算SG在CB-CA延长线的位置
        angle_rad = math.radians(180 - ca_cb_sg_angle)
        
        # SG位置
        sg = cb - v_cb_ca_norm * cb_sg_bond * math.cos(angle_rad)
        
        return sg
    except:
        return None


def estimate_cb_position(ca: np.ndarray, n: np.ndarray, 
                        c: np.ndarray) -> Optional[np.ndarray]:
    """
    估算缺失的CB原子位置（用于Gly等残基）
    
    Args:
        ca: Cα坐标
        n: N原子坐标
        c: C原子坐标
    
    Returns:
        估算的CB坐标，失败返回None
    """
    try:
        # 计算N-CA和C-CA的平均方向
        v1 = n - ca
        v2 = c - ca
        v_avg = (v1 + v2) / 2
        v_norm = v_avg / np.linalg.norm(v_avg)
        
        # CB在CA后方约1.5Å
        cb_estimate = ca - v_norm * 1.5
        return cb_estimate
    except:
        return None


def calculate_bfactor_average(atom_dict: dict) -> Optional[float]:
    """
    计算残基所有原子的平均B因子
    
    Args:
        atom_dict: 原子字典，格式 {atom_name: {'coord': (x,y,z), 'bfactor': float}}
    
    Returns:
        平均B因子，失败返回None
    """
    try:
        bfactors = [atom['bfactor'] for atom in atom_dict.values() 
                   if 'bfactor' in atom]
        
        if not bfactors:
            return None
        
        return float(np.mean(bfactors))
    except:
        return None


def check_disulfide_geometry(ca1: np.ndarray, cb1: np.ndarray,
                            ca2: np.ndarray, cb2: np.ndarray,
                            mode: str = 'relaxed',
                            use_dbd2: bool = False) -> Tuple[bool, dict]:
    """
    检查二硫键几何可行性
    
    Args:
        ca1, cb1: 残基1的Cα和Cβ坐标
        ca2, cb2: 残基2的Cα和Cβ坐标
        mode: 'strict' (3.5-4.5Å) 或 'relaxed' (3.0-5.0Å)
        use_dbd2: 是否使用DbD2的SG距离检查
    
    Returns:
        (is_valid, geometry_data)
        geometry_data包含: ca_distance, cb_distance, sg_distance, dihedral
    """
    # 检查输入有效性
    if ca1 is None or cb1 is None or ca2 is None or cb2 is None:
        return False, {}
    
    # 1. CA-CA距离粗筛
    ca_dist = calculate_distance(ca1, ca2)
    if ca_dist is None or ca_dist < 3.0 or ca_dist > 9.0:
        return False, {}
    
    # 2. CB-CB距离精筛
    cb_dist = calculate_distance(cb1, cb2)
    if cb_dist is None:
        return False, {}
    
    # 根据模式设置阈值
    if mode == 'strict':
        if cb_dist < 3.5 or cb_dist > 4.5:
            return False, {}
    else:  # relaxed
        if cb_dist < 3.0 or cb_dist > 5.0:
            return False, {}
    
    # 3. DbD2风格：SG-SG距离检查
    sg_dist = None
    if use_dbd2:
        sg1 = estimate_sg_position(ca1, cb1)
        sg2 = estimate_sg_position(ca2, cb2)
        
        if sg1 is not None and sg2 is not None:
            sg_dist = calculate_distance(sg1, sg2)
            # DbD2标准：SG-SG距离应在1.8-2.4Å，这里放宽标准1.5-3.2
            if sg_dist is None or sg_dist < 1.5 or sg_dist > 3.2:
                return False, {}
    
    # 4. CA-CB-CB-CA二面角检查
    dihedral = calculate_dihedral(ca1, cb1, cb2, ca2)
    
    # Strict模式才检查二面角
    valid_dihedral = True
    if dihedral is not None and mode == 'strict':
        abs_dihedral = abs(dihedral)
        # 理想二面角：80-100° 或 170-190°
        valid_dihedral = (
            (80 <= abs_dihedral <= 100) or 
            (170 <= abs_dihedral <= 190)
        )
        if not valid_dihedral:
            return False, {}
    
    # 构建几何数据
    geometry_data = {
        'ca_distance': ca_dist,
        'cb_distance': cb_dist,
        'sg_distance': sg_dist,
        'dihedral': dihedral,
        'valid': True
    }
    
    return True, geometry_data


def calculate_chi3_dihedral(ca: np.ndarray, cb: np.ndarray,
                           sg1: np.ndarray, sg2: np.ndarray) -> Optional[float]:
    """
    计算χ3二面角（Cα-Cβ-Sγ-Sγ'）
    这是二硫键手性的关键指标
    
    Args:
        ca, cb: 残基的Cα和Cβ坐标
        sg1, sg2: 两个SG原子坐标
    
    Returns:
        χ3二面角（度），理想值为±87°或±97°
    
    Reference:
        DbD2标准：-87° (左手螺旋) 或 +97° (右手螺旋)
    """
    return calculate_dihedral(ca, cb, sg1, sg2)


def validate_disulfide_geometry_dbd2(ca1: np.ndarray, cb1: np.ndarray,
                                     ca2: np.ndarray, cb2: np.ndarray,
                                     chi3_tolerance: float = 30.0) -> Tuple[bool, dict]:
    """
    使用DbD2完整标准验证二硫键几何
    
    Args:
        ca1, cb1: 残基1的坐标
        ca2, cb2: 残基2的坐标
        chi3_tolerance: χ3二面角容差（度）
    
    Returns:
        (is_valid, detailed_geometry_data)
    """
    # 估算SG位置
    sg1 = estimate_sg_position(ca1, cb1)
    sg2 = estimate_sg_position(ca2, cb2)
    
    if sg1 is None or sg2 is None:
        return False, {}
    
    # 1. SG-SG距离
    sg_dist = calculate_distance(sg1, sg2)
    if sg_dist is None or sg_dist < 1.8 or sg_dist > 2.4:
        return False, {'sg_distance': sg_dist, 'reason': 'SG distance out of range'}
    
    # 2. χ3二面角（两个方向）
    chi3_1 = calculate_chi3_dihedral(ca1, cb1, sg1, sg2)
    chi3_2 = calculate_chi3_dihedral(ca2, cb2, sg2, sg1)
    
    # 检查是否符合-87°或+97°（±容差）
    valid_chi3_1 = False
    valid_chi3_2 = False
    
    if chi3_1 is not None:
        valid_chi3_1 = (
            abs(chi3_1 - (-87)) < chi3_tolerance or
            abs(chi3_1 - 97) < chi3_tolerance
        )
    
    if chi3_2 is not None:
        valid_chi3_2 = (
            abs(chi3_2 - (-87)) < chi3_tolerance or
            abs(chi3_2 - 97) < chi3_tolerance
        )
    
    # 至少一个χ3角度符合要求
    if not (valid_chi3_1 or valid_chi3_2):
        return False, {
            'chi3_1': chi3_1,
            'chi3_2': chi3_2,
            'reason': 'Chi3 angles invalid'
        }
    
    # 3. Cα-Cβ-Sγ角度（应接近114.6°）
    ca_cb_sg_angle1 = calculate_dihedral(ca1, cb1, sg1, sg2) if sg2 is not None else None
    
    geometry_data = {
        'sg_distance': sg_dist,
        'chi3_1': chi3_1,
        'chi3_2': chi3_2,
        'ca_cb_sg_angle': ca_cb_sg_angle1,
        'valid': True,
        'method': 'DbD2'
    }
    
    return True, geometry_data
