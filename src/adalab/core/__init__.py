"""
AdaLab Core - 核心功能模块
"""

from .evaluator import evaluate, evaluate_detailed
from .trainer import TrainingPipeline

__all__ = ['evaluate', 'evaluate_detailed', 'TrainingPipeline']


