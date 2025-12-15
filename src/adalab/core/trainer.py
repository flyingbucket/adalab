"""
训练器模块 - 提供统一的训练流程管理
"""

import os
import json
import joblib
from src.adalab.workflow import train_and_save
from .evaluator import evaluate


class TrainingPipeline:
    """
    训练流程管理器 - 封装端到端的训练-评估-保存流程
    """
    
    def __init__(self, config_path=None):
        """
        初始化训练流程
        
        Parameters
        ----------
        config_path : str, optional
            配置文件路径
        """
        self.config_path = config_path
        self.clf = None
        self.monitor = None
        self.data_prep = None
        self.data = None
        self.paths = None
    
    def run(self):
        """
        执行完整的训练流程
        
        Returns
        -------
        dict
            训练结果和评估分数
        """
        # 1. 训练模型
        print("=" * 60)
        print("开始训练流程".center(60))
        print("=" * 60)
        
        (
            self.clf,
            self.monitor,
            self.data_prep,
            self.data,
            self.paths,
        ) = train_and_save(self.config_path)
        
        X_train, X_test_mnist, y_train, y_test_mnist, noise_idx, clean_idx = self.data
        
        # 2. 评估MNIST测试集
        print("\n" + "=" * 60)
        print("评估MNIST测试集".center(60))
        print("=" * 60)
        
        y_pred_mnist = self.clf.predict(X_test_mnist)
        scores_on_mnist = evaluate(y_true=y_test_mnist, y_pred=y_pred_mnist, 
                                   title="MNIST Test Set")
        
        # 3. 评估课程数据（如果存在）
        scores_on_course = None
        if os.path.exists("data/test_images"):
            print("\n" + "=" * 60)
            print("评估课程测试数据".center(60))
            print("=" * 60)
            
            try:
                X_course, y_course = self.data_prep.prepare_course_data("data/test_images")
                y_pred_course = self.clf.predict(X_course)
                scores_on_course = evaluate(y_true=y_course, y_pred=y_pred_course,
                                           title="Course Test Data")
            except Exception as e:
                print(f"⚠️  无法加载课程数据: {e}")
        
        # 4. 保存评估结果
        scores = {"mnist": scores_on_mnist}
        if scores_on_course:
            scores["course_data"] = scores_on_course
        
        result_dir = self.paths["result_dir"]
        score_path = os.path.join(result_dir, "scores.json")
        with open(score_path, "w") as f:
            json.dump(scores, f, indent=4)
        
        print("\n" + "=" * 60)
        print(f"✅ 评估结果已保存到: {score_path}")
        print("=" * 60)
        
        return {
            "classifier": self.clf,
            "monitor": self.monitor,
            "scores": scores,
            "paths": self.paths,
        }

