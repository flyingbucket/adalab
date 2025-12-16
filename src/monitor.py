"""
兼容性模块 - 用于加载旧版本的joblib文件

旧的 BoostMonitor 对象被序列化时引用了 'src.monitor' 模块
此文件提供向后兼容，使得这些对象可以被正确反序列化
"""

# 从新位置导入
from src.adalab.monitor import BoostMonitor

__all__ = ['BoostMonitor']

