"""
B-factor Analysis Tool
B因子分析工具 - 检查PDB文件的B因子分布和可用性
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from Bio.PDB import PDBParser
import warnings

warnings.filterwarnings('ignore')


class BFactorAnalyzer:
    """B因子分析器"""
    
    def __init__(self, pdb_file: str, quiet: bool = True):
        """
        初始化分析器
        
        Args:
            pdb_file: PDB文件路径
            quiet: 是否静默模式(不打印BioPython警告)
        """
        self.pdb_file = pdb_file
        self.parser = PDBParser(QUIET=quiet)
        self.structure = None
        self.analysis_result = None
        
        # 加载结构
        try:
            self.structure = self.parser.get_structure('protein', pdb_file)
        except Exception as e:
            raise ValueError(f"无法加载PDB文件: {str(e)}")
    
    def analyze(self) -> Dict:
        """
        执行完整的B因子分析
        
        Returns:
            分析结果字典
        """
        # 收集所有B因子
        all_bfactors = []
        residue_bfactors = {}
        
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    res_bfactors = []
                    
                    for atom in residue:
                        bf = atom.get_bfactor()
                        all_bfactors.append(bf)
                        res_bfactors.append(bf)
                    
                    if res_bfactors:
                        res_id = residue.id[1]
                        residue_bfactors[res_id] = np.mean(res_bfactors)
        
        if not all_bfactors:
            raise ValueError("PDB文件中没有B因子数据")
        
        # 统计分析
        all_bfactors = np.array(all_bfactors)
        res_bf_values = np.array(list(residue_bfactors.values()))
        
        # 判断是否归一化
        is_normalized = self._check_normalized(all_bfactors)
        
        # 评估可用性
        usability = self._assess_usability(all_bfactors, is_normalized)
        
        # 构建结果
        self.analysis_result = {
            'pdb_file': self.pdb_file,
            'atom_count': len(all_bfactors),
            'residue_count': len(residue_bfactors),
            
            # 原子级别统计
            'atom_stats': {
                'min': float(np.min(all_bfactors)),
                'max': float(np.max(all_bfactors)),
                'mean': float(np.mean(all_bfactors)),
                'std': float(np.std(all_bfactors)),
                'median': float(np.median(all_bfactors)),
            },
            
            # 残基级别统计
            'residue_stats': {
                'min': float(np.min(res_bf_values)),
                'max': float(np.max(res_bf_values)),
                'mean': float(np.mean(res_bf_values)),
                'std': float(np.std(res_bf_values)),
                'median': float(np.median(res_bf_values)),
            },
            
            # B因子分布
            'bfactor_distribution': {
                'all_bfactors': all_bfactors.tolist(),
                'residue_bfactors': residue_bfactors,
            },
            
            # 判断结果
            'is_normalized': is_normalized,
            'usability': usability,
            'recommendations': self._generate_recommendations(usability, is_normalized)
        }
        
        return self.analysis_result
    
    def _check_normalized(self, bfactors: np.ndarray) -> bool:
        """
        判断B因子是否被归一化
        
        Args:
            bfactors: B因子数组
        
        Returns:
            是否归一化
        """
        b_min = np.min(bfactors)
        b_max = np.max(bfactors)
        b_range = b_max - b_min
        
        # 判断标准:
        # 1. 最大值 < 2.0 → 几乎肯定是归一化的
        # 2. 范围 < 3.0 → 可能是归一化的
        # 3. 最大值 < 10.0 → 可能有问题
        
        if b_max < 2.0:
            return True
        elif b_range < 3.0 and b_max < 5.0:
            return True
        else:
            return False
    
    def _assess_usability(self, bfactors: np.ndarray, is_normalized: bool) -> str:
        """
        评估B因子数据的可用性
        
        Args:
            bfactors: B因子数组
            is_normalized: 是否归一化
        
        Returns:
            可用性等级: 'excellent', 'good', 'poor', 'unusable'
        """
        b_min = np.min(bfactors)
        b_max = np.max(bfactors)
        b_std = np.std(bfactors)
        
        if is_normalized:
            # 归一化后,信息量大幅降低
            if b_std > 0.15:
                return 'poor'  # 有一些变化,但不够
            else:
                return 'unusable'  # 几乎没有变化
        
        else:
            # 真实B因子
            if b_max > 80:
                return 'excellent'  # 范围大,信息丰富
            elif b_max > 50 and b_std > 10:
                return 'good'  # 可用
            elif b_max > 30:
                return 'poor'  # 勉强可用
            else:
                return 'unusable'  # 范围太小
    
    def _generate_recommendations(self, usability: str, is_normalized: bool) -> List[str]:
        """
        生成使用建议
        
        Args:
            usability: 可用性等级
            is_normalized: 是否归一化
        
        Returns:
            建议列表
        """
        recommendations = []
        
        if is_normalized:
            recommendations.append("⚠️ B因子已归一化(范围0-1)")
            recommendations.append("❌ 不建议用于筛选(信息量太少)")
            recommendations.append("💡 建议替代方案:")
            recommendations.append("   - 使用序列距离偏好(15-40)")
            recommendations.append("   - 基于二级结构位置")
            recommendations.append("   - 优先选择loop/turn区域")
        
        else:
            if usability == 'excellent':
                recommendations.append("✅ B因子数据质量优秀")
                recommendations.append("✅ 可以用于筛选热稳定性改造位点")
                recommendations.append("💡 建议甜区: B因子 30-45")
            
            elif usability == 'good':
                recommendations.append("✅ B因子数据可用")
                recommendations.append("⚠️ 变化范围较小,筛选效果可能有限")
                recommendations.append("💡 建议甜区: B因子 25-40")
            
            elif usability == 'poor':
                recommendations.append("⚠️ B因子数据质量较差")
                recommendations.append("⚠️ 可以尝试用于筛选,但效果可能不佳")
                recommendations.append("💡 建议降低B因子权重(<20%)")
            
            else:  # unusable
                recommendations.append("❌ B因子数据不可用")
                recommendations.append("❌ 范围太小或数据有问题")
                recommendations.append("💡 使用其他指标代替")
        
        return recommendations
    
    def print_summary(self):
        """打印分析摘要"""
        if self.analysis_result is None:
            print("请先运行 analyze() 方法")
            return
        
        result = self.analysis_result
        
        print("="*70)
        print("B因子分析报告")
        print("="*70)
        print(f"PDB文件: {result['pdb_file']}")
        print(f"原子数: {result['atom_count']}")
        print(f"残基数: {result['residue_count']}")
        
        print(f"\n{'─'*70}")
        print("原子级别统计:")
        print("─"*70)
        stats = result['atom_stats']
        print(f"  范围: {stats['min']:.2f} - {stats['max']:.2f}")
        print(f"  平均: {stats['mean']:.2f}")
        print(f"  标准差: {stats['std']:.2f}")
        print(f"  中位数: {stats['median']:.2f}")
        
        print(f"\n{'─'*70}")
        print("残基级别统计:")
        print("─"*70)
        stats = result['residue_stats']
        print(f"  范围: {stats['min']:.2f} - {stats['max']:.2f}")
        print(f"  平均: {stats['mean']:.2f}")
        print(f"  标准差: {stats['std']:.2f}")
        print(f"  中位数: {stats['median']:.2f}")
        
        print(f"\n{'='*70}")
        print("评估结果:")
        print("="*70)
        
        if result['is_normalized']:
            print("状态: ⚠️ 已归一化")
        else:
            print("状态: ✅ 真实物理值")
        
        usability_emoji = {
            'excellent': '✅',
            'good': '👍',
            'poor': '⚠️',
            'unusable': '❌'
        }
        emoji = usability_emoji.get(result['usability'], '❓')
        print(f"可用性: {emoji} {result['usability'].upper()}")
        
        print(f"\n{'─'*70}")
        print("建议:")
        print("─"*70)
        for rec in result['recommendations']:
            print(f"  {rec}")
        
        print("="*70)


def analyze_bfactor_distribution(pdb_file: str, verbose: bool = True) -> Dict:
    """
    快速分析PDB文件的B因子分布
    
    Args:
        pdb_file: PDB文件路径
        verbose: 是否打印详细信息
    
    Returns:
        分析结果字典
    
    Example:
        >>> result = analyze_bfactor_distribution('protein.pdb')
        >>> print(f"B因子可用: {result['usability']}")
    """
    analyzer = BFactorAnalyzer(pdb_file)
    result = analyzer.analyze()
    
    if verbose:
        analyzer.print_summary()
    
    return result


def check_pdb_bfactor(pdb_file: str) -> Tuple[bool, str]:
    """
    快速检查PDB文件的B因子是否可用于筛选
    
    Args:
        pdb_file: PDB文件路径
    
    Returns:
        (是否可用, 简短说明)
    
    Example:
        >>> usable, reason = check_pdb_bfactor('protein.pdb')
        >>> if usable:
        >>>     print("可以使用B因子筛选")
        >>> else:
        >>>     print(f"不建议使用: {reason}")
    """
    try:
        analyzer = BFactorAnalyzer(pdb_file, quiet=True)
        result = analyzer.analyze()
        
        is_normalized = result['is_normalized']
        usability = result['usability']
        
        if is_normalized:
            return False, "B因子已归一化,信息量不足"
        
        elif usability in ['excellent', 'good']:
            return True, f"B因子质量{usability},可以使用"
        
        elif usability == 'poor':
            return False, "B因子变化范围小,效果有限"
        
        else:  # unusable
            return False, "B因子数据不可用"
    
    except Exception as e:
        return False, f"分析失败: {str(e)}"


def get_bfactor_for_residues(pdb_file: str, residue_ids: List[int]) -> Dict[int, float]:
    """
    获取指定残基的B因子
    
    Args:
        pdb_file: PDB文件路径
        residue_ids: 残基ID列表
    
    Returns:
        {残基ID: 平均B因子}
    
    Example:
        >>> bfactors = get_bfactor_for_residues('protein.pdb', [49, 68])
        >>> print(f"残基49: {bfactors[49]:.2f}")
        >>> print(f"残基68: {bfactors[68]:.2f}")
    """
    analyzer = BFactorAnalyzer(pdb_file, quiet=True)
    result = analyzer.analyze()
    
    residue_bfactors = result['bfactor_distribution']['residue_bfactors']
    
    output = {}
    for res_id in residue_ids:
        if res_id in residue_bfactors:
            output[res_id] = residue_bfactors[res_id]
        else:
            output[res_id] = None
    
    return output


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python bfactor_checker.py <PDB文件>")
        print("示例: python bfactor_checker.py protein.pdb")
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    
    print(f"\n分析文件: {pdb_file}\n")
    
    # 执行分析
    result = analyze_bfactor_distribution(pdb_file, verbose=True)
    
    # 快速检查
    print("\n" + "="*70)
    usable, reason = check_pdb_bfactor(pdb_file)
    print(f"快速判断: {'✅ 可用' if usable else '❌ 不可用'}")
    print(f"原因: {reason}")
    print("="*70)
