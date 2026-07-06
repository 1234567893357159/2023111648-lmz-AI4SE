"""
步骤四 & 五：SVM 和 Random Forest 模型训练与评估
预测目标：Pull Request 是否最终 Merge
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    roc_curve, auc,
)

import config


# ============================================================
# BaseTrainer：共享数据加载 + 预处理 + 评估工具
# ============================================================
class BaseTrainer:
    def __init__(self):
        self.feature_csv = os.path.join(config.FEATURES_DIR, "human_features.csv")
        self.scaler_path = os.path.join(config.MODELS_DIR, "scaler.joblib")

        self.feature_names = None
        self.scaler = None
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None

    def _load_and_merge_data(self):
        df = pd.read_csv(self.feature_csv)
        self.feature_names = [c for c in df.columns if c not in ("pr_id", "repo")]

        labels = []
        for _, row in df.iterrows():
            pr_id = row["pr_id"]
            repo = row["repo"]
            pulls_file = os.path.join(config.HUMAN_DIR, f"{repo}_pulls.json")
            merged = 0
            if os.path.exists(pulls_file):
                with open(pulls_file, "r", encoding="utf-8") as f:
                    pulls = json.load(f)
                for pr in pulls:
                    if pr.get("pr_id") == pr_id:
                        merged = 1 if pr.get("merged") else 0
                        break
            labels.append(merged)

        X = df[self.feature_names].copy()
        y = np.array(labels)

        print(f"数据加载完成: {X.shape[0]} 条样本, {X.shape[1]} 维特征")
        print(f"  标签分布: merged={sum(y)}, not_merged={len(y)-sum(y)}")
        return X, y

    def _preprocess(self, X, y):
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        joblib.dump(self.scaler, self.scaler_path)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def _ensure_data_loaded(self):
        if self._X_train is not None:
            return
        X, y = self._load_and_merge_data()
        self._X_train, self._X_test, self._y_train, self._y_test = self._preprocess(X, y)

    def _evaluate_model(self, model, X_test, y_test, name):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n{'='*60}")
        print(f"{name} 评估结果")
        print(f"{'='*60}")
        print(f"  Accuracy:   {acc:.4f}")
        print(f"  Precision:  {prec:.4f}")
        print(f"  Recall:     {rec:.4f}")
        print(f"  F1-score:   {f1:.4f}")
        print(f"  ROC-AUC:    {roc_auc:.4f}")
        print(f"  混淆矩阵:")
        print(f"    TN={cm[0][0]}, FP={cm[0][1]}")
        print(f"    FN={cm[1][0]}, TP={cm[1][1]}")

        return {
            "name": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "roc_auc": roc_auc,
            "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba,
        }

    def _plot_confusion_matrix(self, result, name):
        cm = confusion_matrix(result["y_test"], result["y_pred"])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                        fontsize=16, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Not Merged", "Merged"])
        ax.set_yticklabels(["Not Merged", "Merged"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name} Confusion Matrix")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        save_path = os.path.join(config.RESULTS_DIR, f"confusion_matrix_{name.lower().replace(' ', '_')}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()


# ============================================================
# SVMTrainer：SVM 训练 + 评估
# ============================================================
class SVMTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(config.MODELS_DIR, "svm_model.joblib")
        self.model = None

    def train(self):
        print("=" * 60)
        print("训练 SVM (Support Vector Machine)")
        print("=" * 60)
        self._ensure_data_loaded()

        param_grid = {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto", 0.01, 0.1],
            "kernel": ["rbf"],
        }
        svm = SVC(probability=True, random_state=42)
        grid = GridSearchCV(svm, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)
        grid.fit(self._X_train, self._y_train)
        print(f"  最佳参数: {grid.best_params_}")
        print(f"  最佳 CV F1: {grid.best_score_:.4f}")

        self.model = grid.best_estimator_
        joblib.dump(self.model, self.model_path)
        print("SVM 模型已保存到:", self.model_path)

    def evaluate(self):
        print("=" * 60)
        print("评估 SVM 模型")
        print("=" * 60)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"SVM 模型不存在: {self.model_path}，请先运行 train()")
        self.model = joblib.load(self.model_path)
        self._ensure_data_loaded()
        result = self._evaluate_model(self.model, self._X_test, self._y_test, "SVM")
        self._plot_confusion_matrix(result, "SVM")
        return result


# ============================================================
# RFTrainer：Random Forest 训练 + 评估
# ============================================================
class RFTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(config.MODELS_DIR, "rf_model.joblib")
        self.model = None

    def train(self):
        print("=" * 60)
        print("训练 Random Forest")
        print("=" * 60)
        self._ensure_data_loaded()

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        }
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid = GridSearchCV(rf, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)
        grid.fit(self._X_train, self._y_train)
        print(f"  最佳参数: {grid.best_params_}")
        print(f"  最佳 CV F1: {grid.best_score_:.4f}")

        self.model = grid.best_estimator_
        joblib.dump(self.model, self.model_path)
        print("Random Forest 模型已保存到:", self.model_path)

    def evaluate(self):
        print("=" * 60)
        print("评估 Random Forest 模型")
        print("=" * 60)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"RF 模型不存在: {self.model_path}，请先运行 train()")
        self.model = joblib.load(self.model_path)
        self._ensure_data_loaded()
        result = self._evaluate_model(self.model, self._X_test, self._y_test, "Random Forest")
        self._plot_confusion_matrix(result, "Random Forest")
        self._plot_feature_importance()
        return result

    def _plot_feature_importance(self):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        top_features = [self.feature_names[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(range(len(top_features)), top_importances[::-1], color="steelblue")
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels([top_features[i] for i in range(len(top_features) - 1, -1, -1)],
                           fontsize=8)
        ax.set_xlabel("Feature Importance")
        ax.set_title("Random Forest Feature Importance (Top 20)")
        plt.tight_layout()
        save_path = os.path.join(config.RESULTS_DIR, "feature_importance_rf.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"\n{'='*60}")
        print("特征重要性 Top 10 (Random Forest)")
        print(f"{'='*60}")
        for rank, (idx, imp) in enumerate(zip(indices[:10], top_importances[:10]), 1):
            print(f"  {rank:2d}. {self.feature_names[idx]:<50s} {imp:.4f}")


# ============================================================
# ModelTrainer：统一入口（兼容旧接口）
# ============================================================
class ModelTrainer:
    def __init__(self):
        self.svm = SVMTrainer()
        self.rf = RFTrainer()
        self.report_path = os.path.join(config.RESULTS_DIR, "evaluation_report.txt")

    def train_svm(self):
        self.svm.train()

    def train_rf(self):
        self.rf.train()

    def evaluate_svm(self):
        return self.svm.evaluate()

    def evaluate_rf(self):
        return self.rf.evaluate()

    def evaluate_both(self):
        print("=" * 60)
        print("模型对比评估")
        print("=" * 60)

        self.svm._ensure_data_loaded()
        self.rf._X_train = self.svm._X_train
        self.rf._X_test = self.svm._X_test
        self.rf._y_train = self.svm._y_train
        self.rf._y_test = self.svm._y_test
        self.rf.feature_names = self.svm.feature_names

        if not os.path.exists(self.svm.model_path):
            raise FileNotFoundError(f"SVM 模型不存在: {self.svm.model_path}")
        if not os.path.exists(self.rf.model_path):
            raise FileNotFoundError(f"RF 模型不存在: {self.rf.model_path}")

        self.svm.model = joblib.load(self.svm.model_path)
        self.rf.model = joblib.load(self.rf.model_path)

        svm_result = self.svm._evaluate_model(self.svm.model, self.svm._X_test, self.svm._y_test, "SVM")
        rf_result = self.rf._evaluate_model(self.rf.model, self.rf._X_test, self.rf._y_test, "Random Forest")

        self.svm._plot_confusion_matrix(svm_result, "SVM")
        self.rf._plot_confusion_matrix(rf_result, "Random Forest")

        self._plot_roc_curve([svm_result, rf_result])
        self.rf._plot_feature_importance()
        self._save_report(svm_result, rf_result)

        print(f"\n结果目录: {config.RESULTS_DIR}")
        return svm_result, rf_result

    def _plot_roc_curve(self, results):
        fig, ax = plt.subplots(figsize=(6, 5))
        for r in results:
            fpr, tpr, _ = roc_curve(r["y_test"], r["y_proba"])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{r['name']} (AUC={roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        plt.tight_layout()
        save_path = os.path.join(config.RESULTS_DIR, "roc_curve.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

    def _save_report(self, svm_result, rf_result):
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write("模型评估报告\n")
            f.write("=" * 60 + "\n\n")
            for r in [svm_result, rf_result]:
                f.write(f"{r['name']}:\n")
                f.write(f"  Accuracy:   {r['accuracy']:.4f}\n")
                f.write(f"  Precision:  {r['precision']:.4f}\n")
                f.write(f"  Recall:     {r['recall']:.4f}\n")
                f.write(f"  F1-score:   {r['f1']:.4f}\n")
                f.write(f"  ROC-AUC:    {r['roc_auc']:.4f}\n\n")

    def train_and_evaluate(self):
        self.train_svm()
        self.train_rf()
        return self.evaluate_both()

    def run_all(self):
        return self.train_and_evaluate()


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_all()