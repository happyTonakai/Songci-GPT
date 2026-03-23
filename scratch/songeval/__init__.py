"""
SongEval v2.0 - 宋词格律自动提取与多维度评估系统

本模块提供：
1. RegistryBuilder: 从训练语料中自动逆向工程出格律库
2. Evaluator: 对模型生成的宋词进行格律符合度评估
"""

from .evaluator import Evaluator
from .registry_builder import RegistryBuilder

__all__ = ["RegistryBuilder", "Evaluator"]
