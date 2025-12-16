"""
Candidate Scoring System
候选评分系统 - 两阶段智能评分
"""

import numpy as np
from typing import Optional, Dict
from enum import Enum


class ScoringStrategy(Enum):
    """评分策略"""
    CONSERVATIVE = "conservative"  # 保守(ΔΔG权重70%)
    BALANCED = "balanced"          # 平衡(ΔΔG权重50%)
    AGGRESSIVE = "aggressive"      # 激进(优先高B因子区域)


class CandidateScorer:
    """候选评分器"""
    
    def __init__(self, use_bfactor: bool = True):
        """
        初始化评分器
        
        Args:
            use_bfactor: 是否使用B因子进行评分
        """
        self.use_bfactor = use_bfactor
    
    def quick_score(self, candidate, config: Optional[Dict] = None) -> float:
        """
        快速评分(阶段1 - 不需要FoldX)
        
        用于预筛选,决定哪些候选进入FoldX计算
        
        Args:
            candidate: DisulfideCandidate对象
            config: 配置参数
        
        Returns:
            评分(0-1,越高越好)
        """
        score = 0.0
        
        # 默认配置
        if config is None:
            config = {
                'cb_weight': 0.30,
                'seq_weight': 0.40,
                'dihedral_weight': 0.20,
                'bfactor_weight': 0.10
            }
        
        # 1. CB距离得分(30%)
        cb_score = self._score_cb_distance(candidate.cb_distance)
        score += cb_score * config['cb_weight']
        
        # 2. 序列距离得分(40%)
        seq_score = self._score_sequence_separation(candidate.seq_separation)
        score += seq_score * config['seq_weight']
        
        # 3. 二面角得分(20%)
        dihedral_score = self._score_dihedral(candidate.dihedral)
        score += dihedral_score * config['dihedral_weight']
        
        # 4. B因子得分(10%) - 可选
        if self.use_bfactor and candidate.res1_bfactor is not None:
            bfactor_score = self._score_bfactor(
                candidate.res1_bfactor,
                candidate.res2_bfactor
            )
            score += bfactor_score * config['bfactor_weight']
        else:
            # B因子不可用,权重分配给其他项
            redistributed = config['bfactor_weight'] / 3
            score += cb_score * redistributed
            score += seq_score * redistributed
            score += dihedral_score * redistributed
        
        return min(1.0, max(0.0, score))
    
    def final_score_with_ddg(self, candidate, 
                            strategy: str = 'balanced') -> float:
        """
        最终评分(阶段2 - 基于FoldX ΔΔG)
        
        Args:
            candidate: DisulfideCandidate对象(必须有ddg值)
            strategy: 评分策略 ('conservative', 'balanced', 'aggressive')
        
        Returns:
            最终评分(0-1,越高越好)
        """
        if candidate.ddg is None:
            # 没有ΔΔG,使用快速评分的一半
            return self.quick_score(candidate) * 0.5
        
        # 根据策略选择权重
        if strategy == 'conservative':
            weights = {
                'ddg': 0.70,
                'cb': 0.15,
                'seq': 0.10,
                'bfactor': 0.05
            }
        elif strategy == 'balanced':
            weights = {
                'ddg': 0.50,
                'cb': 0.20,
                'seq': 0.20,
                'bfactor': 0.10
            }
        elif strategy == 'aggressive':
            weights = {
                'ddg': 0.40,
                'cb': 0.15,
                'seq': 0.15,
                'bfactor': 0.30
            }
        else:
            weights = {'ddg': 0.50, 'cb': 0.20, 'seq': 0.20, 'bfactor': 0.10}
        
        score = 0.0
        
        # 1. ΔΔG得分
        ddg_score = self._score_ddg(candidate.ddg)
        score += ddg_score * weights['ddg']
        
        # 2. CB距离得分
        cb_score = self._score_cb_distance(candidate.cb_distance)
        score += cb_score * weights['cb']
        
        # 3. 序列距离得分
        seq_score = self._score_sequence_separation(candidate.seq_separation)
        score += seq_score * weights['seq']
        
        # 4. B因子得分
        if self.use_bfactor and candidate.res1_bfactor is not None:
            bfactor_score = self._score_bfactor(
                candidate.res1_bfactor,
                candidate.res2_bfactor
            )
            score += bfactor_score * weights['bfactor']
        else:
            # 重新分配权重
            redistributed = weights['bfactor'] / 3
            score += ddg_score * redistributed
            score += cb_score * redistributed
            score += seq_score * redistributed
        
        return min(1.0, max(0.0, score))
    
    # ============================================================
    # 各项指标的评分函数
    # ============================================================
    
    def _score_cb_distance(self, cb_distance: float) -> float:
        """
        CB距离评分
        
        理想值: 4.0Å
        可接受范围: 3.5-4.5Å
        """
        if cb_distance is None:
            return 0.5
        
        deviation = abs(cb_distance - 4.0)
        
        if deviation < 0.1:
            return 1.0  # 完美
        elif deviation < 0.3:
            return 0.9  # 优秀
        elif deviation < 0.5:
            return 0.7  # 良好
        else:
            # 线性衰减
            return max(0.0, 0.7 - (deviation - 0.5) * 0.5)
    
    def _score_sequence_separation(self, seq_separation: int) -> float:
        """
        序列距离评分
        
        甜区: 15-35 (中等距离)
        可接受: 10-50
        """
        if seq_separation is None:
            return 0.5
        
        # 甜区
        if 15 <= seq_separation <= 35:
            return 1.0
        
        # 次优区
        elif 10 <= seq_separation < 15:
            return 0.7 + (seq_separation - 10) / 5 * 0.3
        elif 35 < seq_separation <= 50:
            return 1.0 - (seq_separation - 35) / 15 * 0.3
        
        # 边缘区
        elif 5 <= seq_separation < 10:
            return 0.4 + (seq_separation - 5) / 5 * 0.3
        elif 50 < seq_separation <= 70:
            return 0.7 - (seq_separation - 50) / 20 * 0.3
        
        # 不推荐区
        else:
            return 0.2
    
    def _score_dihedral(self, dihedral: Optional[float]) -> float:
        """
        二面角评分
        
        理想: ±90° 或 ±180°
        可接受: 60-120° 或 150-210°
        """
        if dihedral is None:
            return 0.5  # 没有信息,中性评分
        
        abs_dihedral = abs(dihedral)
        
        # 理想范围
        if 80 <= abs_dihedral <= 100:
            return 1.0  # ±90°
        elif 170 <= abs_dihedral <= 190:
            return 0.95  # ±180°
        
        # 可接受范围
        elif 60 <= abs_dihedral < 80:
            return 0.7 + (abs_dihedral - 60) / 20 * 0.3
        elif 100 < abs_dihedral <= 120:
            return 1.0 - (abs_dihedral - 100) / 20 * 0.3
        elif 150 <= abs_dihedral < 170:
            return 0.7 + (abs_dihedral - 150) / 20 * 0.25
        elif 190 < abs_dihedral <= 210:
            return 0.95 - (abs_dihedral - 190) / 20 * 0.25
        
        # 不理想范围
        else:
            return 0.3
    
    def _score_bfactor(self, bf1: float, bf2: float) -> float:
        """
        B因子评分
        
        目标: 选择"略高但不太乱"的区域
        甜区: 平均30-45
        """
        if bf1 is None or bf2 is None:
            return 0.5
        
        # 计算平均值
        avg_bf = (bf1 + bf2) / 2
        
        # 计算差异(避免一个太高一个太低)
        diff_bf = abs(bf1 - bf2)
        
        score = 0.0
        
        # 1. 平均B因子得分(70%)
        if 30 <= avg_bf <= 45:
            bf_score = 1.0  # 甜区
        elif 25 <= avg_bf < 30 or 45 < avg_bf <= 50:
            bf_score = 0.7
        elif 20 <= avg_bf < 25 or 50 < avg_bf <= 55:
            bf_score = 0.4
        elif avg_bf < 20:
            bf_score = 0.2  # 太稳定,提升空间小
        elif avg_bf > 55:
            bf_score = 0.1  # 太无序,风险高
        else:
            bf_score = 0.0
        
        score += bf_score * 0.7
        
        # 2. 一致性得分(30%)
        if diff_bf < 10:
            consistency_score = 1.0
        elif diff_bf < 20:
            consistency_score = 0.6
        else:
            consistency_score = 0.3
        
        score += consistency_score * 0.3
        
        # 3. 极端值惩罚
        if bf1 > 60 or bf2 > 60:
            score = 0.0  # 直接淘汰
        elif bf1 < 15 or bf2 < 15:
            score *= 0.5  # 降权
        
        return score
    
    def _score_ddg(self, ddg: float) -> float:
        """
        ΔΔG评分
        
        原则: 越低越好(越不"别扭")
        优秀: < 2.0
        良好: 2.0 - 5.0
        可接受: 5.0 - 8.0
        不推荐: > 8.0
        """
        if ddg is None:
            return 0.0
        
        if ddg < 0:
            return 1.0  # 负值最好(稳定化)
        elif ddg < 1.0:
            return 0.95
        elif ddg < 2.0:
            return 0.9 - (ddg - 1.0) * 0.1
        elif ddg < 5.0:
            return 0.8 - (ddg - 2.0) / 3.0 * 0.3
        elif ddg < 8.0:
            return 0.5 - (ddg - 5.0) / 3.0 * 0.3
        else:
            # 线性衰减,但不低于0.1
            return max(0.1, 0.2 - (ddg - 8.0) / 10.0 * 0.1)
    
    # ============================================================
    # 批量评分和排序
    # ============================================================
    
    def rank_candidates(self, candidates, mode='pre_foldx', 
                       strategy='balanced'):
        """
        对候选进行排序
        
        Args:
            candidates: 候选列表
            mode: 'pre_foldx' 或 'post_foldx'
            strategy: 评分策略(仅post_foldx模式)
        
        Returns:
            排序后的候选列表
        """
        if mode == 'pre_foldx':
            # 使用快速评分
            for cand in candidates:
                if not hasattr(cand, 'quick_score'):
                    cand.quick_score = self.quick_score(cand)
            
            return sorted(candidates, 
                         key=lambda x: x.quick_score, 
                         reverse=True)
        
        else:  # post_foldx
            # 使用最终评分
            for cand in candidates:
                if not hasattr(cand, 'final_score'):
                    cand.final_score = self.final_score_with_ddg(cand, strategy)
            
            return sorted(candidates,
                         key=lambda x: x.final_score,
                         reverse=True)


def calculate_score_breakdown(candidate, scorer: CandidateScorer, 
                              mode='post_foldx', strategy='balanced') -> Dict:
    """
    计算评分的详细分解
    
    用于报告生成,显示每个指标的贡献
    
    Args:
        candidate: 候选对象
        scorer: 评分器
        mode: 'pre_foldx' 或 'post_foldx'
        strategy: 评分策略
    
    Returns:
        评分分解字典
    """
    breakdown = {}
    
    if mode == 'pre_foldx':
        breakdown['cb_distance'] = {
            'value': candidate.cb_distance,
            'score': scorer._score_cb_distance(candidate.cb_distance),
            'weight': 0.30
        }
        breakdown['seq_separation'] = {
            'value': candidate.seq_separation,
            'score': scorer._score_sequence_separation(candidate.seq_separation),
            'weight': 0.40
        }
        breakdown['dihedral'] = {
            'value': candidate.dihedral,
            'score': scorer._score_dihedral(candidate.dihedral),
            'weight': 0.20
        }
        if scorer.use_bfactor and candidate.res1_bfactor is not None:
            breakdown['bfactor'] = {
                'value': (candidate.res1_bfactor + candidate.res2_bfactor) / 2,
                'score': scorer._score_bfactor(candidate.res1_bfactor, 
                                               candidate.res2_bfactor),
                'weight': 0.10
            }
    
    else:  # post_foldx
        # 根据策略设置权重
        if strategy == 'conservative':
            weights = {'ddg': 0.70, 'cb': 0.15, 'seq': 0.10, 'bfactor': 0.05}
        elif strategy == 'balanced':
            weights = {'ddg': 0.50, 'cb': 0.20, 'seq': 0.20, 'bfactor': 0.10}
        else:  # aggressive
            weights = {'ddg': 0.40, 'cb': 0.15, 'seq': 0.15, 'bfactor': 0.30}
        
        breakdown['ddg'] = {
            'value': candidate.ddg,
            'score': scorer._score_ddg(candidate.ddg),
            'weight': weights['ddg']
        }
        breakdown['cb_distance'] = {
            'value': candidate.cb_distance,
            'score': scorer._score_cb_distance(candidate.cb_distance),
            'weight': weights['cb']
        }
        breakdown['seq_separation'] = {
            'value': candidate.seq_separation,
            'score': scorer._score_sequence_separation(candidate.seq_separation),
            'weight': weights['seq']
        }
        if scorer.use_bfactor and candidate.res1_bfactor is not None:
            breakdown['bfactor'] = {
                'value': (candidate.res1_bfactor + candidate.res2_bfactor) / 2,
                'score': scorer._score_bfactor(candidate.res1_bfactor,
                                               candidate.res2_bfactor),
                'weight': weights['bfactor']
            }
    
    # 计算总分
    total = sum(item['score'] * item['weight'] 
                for item in breakdown.values())
    breakdown['total_score'] = total
    
    return breakdown


if __name__ == "__main__":
    print("Candidate Scoring System - 测试模式")
    print("="*70)
    
    # 模拟候选对象
    class MockCandidate:
        def __init__(self):
            self.cb_distance = 3.99
            self.seq_separation = 25
            self.dihedral = 92.0
            self.res1_bfactor = 35.0
            self.res2_bfactor = 38.0
            self.ddg = 1.2
    
    cand = MockCandidate()
    scorer = CandidateScorer(use_bfactor=True)
    
    # 测试快速评分
    quick = scorer.quick_score(cand)
    print(f"快速评分: {quick:.3f}")
    
    # 测试最终评分
    for strategy in ['conservative', 'balanced', 'aggressive']:
        final = scorer.final_score_with_ddg(cand, strategy)
        print(f"最终评分({strategy}): {final:.3f}")
    
    # 测试评分分解
    print("\n评分分解(balanced策略):")
    breakdown = calculate_score_breakdown(cand, scorer, 'post_foldx', 'balanced')
    for key, value in breakdown.items():
        if key != 'total_score':
            print(f"  {key}: {value['score']:.2f} × {value['weight']:.2f} = {value['score']*value['weight']:.3f}")
    print(f"  总分: {breakdown['total_score']:.3f}")
