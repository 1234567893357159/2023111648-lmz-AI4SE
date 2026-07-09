"""
评估器模块
计算分类指标并绘制可视化图表
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

import sys
sys.path.insert(0, "..")
import config

sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class Evaluator:
    """模型评估器，计算指标并生成图表"""

    def __init__(self):
        self.results = {}
        self.y_test = None
        self.predictions = {}
        self.probabilities = {}

    def set_test_data(self, y_test: np.ndarray):
        """设置测试集标签"""
        self.y_test = y_test

    def add_model(self, name: str, y_pred: np.ndarray, y_proba: np.ndarray = None):
        """添加模型预测结果"""
        self.predictions[name] = y_pred
        if y_proba is not None:
            self.probabilities[name] = y_proba

    def compute_metrics(self, name: str) -> dict:
        """计算单个模型的所有指标"""
        y_pred = self.predictions[name]
        y_true = self.y_test
        y_proba = self.probabilities.get(name)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

        self.results[name] = metrics
        return metrics

    def compute_all(self):
        """计算所有模型的指标"""
        for name in self.predictions:
            self.compute_metrics(name)
        return self.results

    def plot_confusion_matrix(self, name: str):
        """绘制混淆矩阵"""
        cm = confusion_matrix(self.y_test, self.predictions[name])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Merged", "Merged"],
            yticklabels=["Not Merged", "Merged"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {name}")
        path = os.path.join(config.PLOTS_DIR, f"confusion_matrix_{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  混淆矩阵已保存: {path}")

    def plot_confusion_matrices_all(self):
        """绘制所有模型的混淆矩阵"""
        for name in self.predictions:
            self.plot_confusion_matrix(name)

    def plot_roc_curves(self):
        """绘制所有模型的 ROC 曲线对比"""
        if not self.probabilities:
            print("  无概率预测结果，跳过 ROC 曲线")
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, y_proba in self.probabilities.items():
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            auc = roc_auc_score(self.y_test, y_proba)
            ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves Comparison")
        ax.legend(loc="lower right")
        path = os.path.join(config.PLOTS_DIR, "roc_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ROC 曲线已保存: {path}")

    def plot_metrics_comparison(self):
        """绘制模型指标对比柱状图"""
        if not self.results:
            self.compute_all()

        model_names = list(self.results.keys())
        metric_names = [m for m in config.METRICS if m in next(iter(self.results.values()))]

        x = np.arange(len(model_names))
        width = 0.15
        n_metrics = len(metric_names)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

        for i, metric in enumerate(metric_names):
            values = [self.results[name][metric] for name in model_names]
            offset = (i - n_metrics / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=metric.capitalize(), color=colors[i])
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.legend(loc="lower right")
        path = os.path.join(config.PLOTS_DIR, "metrics_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  指标对比图已保存: {path}")

    def plot_feature_importance(self, importance: np.ndarray, top_n: int = 20):
        """绘制随机森林特征重要性"""
        n_features = len(importance)
        if n_features > top_n:
            indices = np.argsort(importance)[-top_n:]
        else:
            indices = np.argsort(importance)

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(indices))
        ax.barh(y_pos, importance[indices])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Dim_{i}" for i in indices])
        ax.set_xlabel("Importance")
        ax.set_title(f"Random Forest - Top {top_n} Feature Importance")
        path = os.path.join(config.PLOTS_DIR, "feature_importance.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  特征重要性已保存: {path}")

    def plot_loss_curve(self, loss_curve: list):
        """绘制 MLP 训练损失曲线"""
        if loss_curve is None or len(loss_curve) == 0:
            print("  损失曲线数据为空，跳过绘制（模型可能是加载的而非训练的）")
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(loss_curve, color="#e74c3c", lw=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("MLP Training Loss Curve")
        ax.grid(True, alpha=0.3)
        path = os.path.join(config.PLOTS_DIR, "mlp_loss_curve.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  损失曲线已保存: {path}")

    def print_summary(self):
        """打印评估结果汇总"""
        if not self.results:
            self.compute_all()

        metric_names = [m for m in config.METRICS if m in next(iter(self.results.values()))]
        header = f"{'Model':<18}" + "".join(f"{m.capitalize():>10}" for m in metric_names)
        sep = "-" * len(header)

        print("\n" + "=" * 60)
        print("评估结果汇总")
        print("=" * 60)
        print(header)
        print(sep)
        for name, metrics in self.results.items():
            row = f"{name:<18}" + "".join(f"{metrics[m]:>10.4f}" for m in metric_names)
            print(row)
        print("=" * 60)

    def save_results(self):
        """保存评估结果到 JSON"""
        path = os.path.join(config.RESULTS_DIR, "classification_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"评估结果已保存: {path}")