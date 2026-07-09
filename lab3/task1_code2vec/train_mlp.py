"""
MLP 训练器
"""

import os
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier

import sys
sys.path.insert(0, "..")
import config


class MLPTrainer:
    """多层感知机 (MLP) 分类器训练器"""

    def __init__(self):
        self.model = None
        self.loss_curve = None
        self.model_path = os.path.join(config.MODELS_DIR, "mlp.pkl")

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """训练 MLP 模型"""
        self.model = MLPClassifier(
            hidden_layer_sizes=config.MLP_HIDDEN_LAYERS,
            max_iter=config.MLP_MAX_ITER,
            learning_rate_init=config.MLP_LEARNING_RATE,
            batch_size=config.MLP_BATCH_SIZE,
            early_stopping=config.MLP_EARLY_STOPPING,
            validation_fraction=config.MLP_VALIDATION_FRACTION,
            random_state=config.RANDOM_SEED,
            verbose=False,
        )
        self.model.fit(X_train, y_train)

        self.loss_curve = self.model.loss_curve_
        train_acc = self.model.score(X_train, y_train)
        print(f"  训练集准确率: {train_acc:.4f}")
        print(f"  迭代次数: {self.model.n_iter_}")

    def get_loss_curve(self):
        """获取训练损失曲线"""
        return self.loss_curve

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        return self.model.predict_proba(X)[:, 1]

    def save(self):
        """保存模型"""
        joblib.dump({"model": self.model, "loss_curve": self.loss_curve}, self.model_path)
        print(f"  模型已保存到: {self.model_path}")

    def load(self):
        """加载模型"""
        data = joblib.load(self.model_path)
        if isinstance(data, dict):
            self.model = data["model"]
            self.loss_curve = data.get("loss_curve")
        else:
            self.model = data
            self.loss_curve = None
        print(f"  模型已加载: {self.model_path}")