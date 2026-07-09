"""
随机森林训练器
"""

import os
import numpy as np
import torch
import joblib
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.insert(0, "..")
import config


class RandomForestTrainer:
    """随机森林分类器训练器"""

    def __init__(self):
        self.model = None
        self.model_path = os.path.join(config.MODELS_DIR, "random_forest.pkl")

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """训练随机森林模型"""
        self.model = RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
            class_weight=config.RF_CLASS_WEIGHT,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        train_acc = self.model.score(X_train, y_train)
        print(f"  训练集准确率: {train_acc:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        return self.model.feature_importances_

    def save(self):
        """保存模型"""
        joblib.dump(self.model, self.model_path)
        print(f"  模型已保存到: {self.model_path}")

    def load(self):
        """加载模型"""
        self.model = joblib.load(self.model_path)
        print(f"  模型已加载: {self.model_path}")