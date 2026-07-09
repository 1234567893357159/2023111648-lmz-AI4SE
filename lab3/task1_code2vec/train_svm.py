"""
SVM 训练器
"""

import os
import numpy as np
import joblib
from sklearn.svm import SVC

import sys
sys.path.insert(0, "..")
import config


class SVMTrainer:
    """SVM 分类器训练器"""

    def __init__(self):
        self.model = None
        self.model_path = os.path.join(config.MODELS_DIR, "svm.pkl")

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """训练 SVM 模型"""
        self.model = SVC(
            C=config.SVM_C,
            kernel=config.SVM_KERNEL,
            gamma=config.SVM_GAMMA,
            class_weight=config.SVM_CLASS_WEIGHT,
            probability=True,
            random_state=config.RANDOM_SEED,
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

    def save(self):
        """保存模型"""
        joblib.dump(self.model, self.model_path)
        print(f"  模型已保存到: {self.model_path}")

    def load(self):
        """加载模型"""
        self.model = joblib.load(self.model_path)
        print(f"  模型已加载: {self.model_path}")