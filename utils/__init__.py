"""
Disulfide Bond Predictor - Utility Modules
辅助工具模块
"""

from .bfactor_checker import (
    analyze_bfactor_distribution,
    check_pdb_bfactor,
    BFactorAnalyzer
)

from .scoring import (
    CandidateScorer,
    ScoringStrategy
)

from .report_generator import (
    generate_enhanced_report,
    generate_layered_recommendations
)

__version__ = "1.0.0"

__all__ = [
    # B因子分析
    'analyze_bfactor_distribution',
    'check_pdb_bfactor',
    'BFactorAnalyzer',
    
    # 评分系统
    'CandidateScorer',
    'ScoringStrategy',
    
    # 报告生成
    'generate_enhanced_report',
    'generate_layered_recommendations',
]
