"""
步骤二：传统机器学习模型预测
加载 lab2 训练好的 SVM / Random Forest 模型，对 AI 代码特征进行预测和评估
"""

import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


class MLPredictor:
    """传统机器学习模型预测器"""

    def __init__(self):
        self.features_df = None
        self.X = None
        self.y = None
        self.scaler = None
        self.svm_model = None
        self.rf_model = None
        self.results = {}

    def load_features(self) -> pd.DataFrame:
        """加载 AI 代码特征"""
        print(f"加载 AI 代码特征: {config.AI_FEATURES_PATH}")
        self.features_df = pd.read_csv(config.AI_FEATURES_PATH)
        print(f"  特征矩阵: {self.features_df.shape}")
        return self.features_df

    def load_labels(self):
        """从 AI PR 数据中加载标签"""
        labels = []
        for _, row in self.features_df.iterrows():
            pr_id = row["pr_id"]
            repo = row["repo"]
            pulls_file = os.path.join(config.AI_PULLS_DIR, f"{repo}_pulls.json")
            merged = 0
            if os.path.exists(pulls_file):
                with open(pulls_file, "r", encoding="utf-8") as f:
                    pulls = json.load(f)
                for pr in pulls:
                    if pr.get("pr_id") == pr_id:
                        merged = 1 if pr.get("merged") else 0
                        break
            labels.append(merged)

        self.X = self.features_df.drop(columns=["pr_id", "repo"]).copy()
        self.y = np.array(labels)

        print(f"标签加载完成: {len(self.y)} 条")
        print(f"  标签分布: merged={sum(self.y)}, not_merged={len(self.y)-sum(self.y)}")
        return self.X, self.y

    def load_models(self):
        """加载 lab2 训练好的模型"""
        print("\n加载 lab2 训练好的模型...")

        self.scaler = joblib.load(config.LAB2_SCALER_PATH)
        print(f"  Scaler 已加载: {config.LAB2_SCALER_PATH}")

        self.svm_model = joblib.load(config.LAB2_SVM_MODEL_PATH)
        print(f"  SVM 模型已加载: {config.LAB2_SVM_MODEL_PATH}")

        self.rf_model = joblib.load(config.LAB2_RF_MODEL_PATH)
        print(f"  Random Forest 模型已加载: {config.LAB2_RF_MODEL_PATH}")

    def preprocess(self):
        """预处理特征（与 lab2 保持一致）"""
        self.X = self.X.fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(self.X)
        print(f"  特征标准化完成: {X_scaled.shape}")
        return X_scaled

    def predict_and_evaluate(self):
        """预测并评估两个模型"""
        print("\n" + "=" * 60)
        print("AI 代码预测与评估")
        print("=" * 60)

        X_scaled = self.preprocess()

        results = {
            "data_info": {
                "total_samples": len(self.y),
                "merged_count": int(sum(self.y)),
                "not_merged_count": int(len(self.y) - sum(self.y)),
                "merged_ratio": round(float(sum(self.y) / len(self.y)), 4) if len(self.y) > 0 else 0,
                "feature_dim": self.X.shape[1],
            }
        }

        for model_name, model in [("SVM", self.svm_model), ("Random Forest", self.rf_model)]:
            print(f"\n{'='*60}")
            print(f"评估 {model_name}")
            print(f"{'='*60}")

            y_pred = model.predict(X_scaled)
            y_proba = model.predict_proba(X_scaled)[:, 1]

            acc = accuracy_score(self.y, y_pred)
            prec = precision_score(self.y, y_pred, zero_division=0)
            rec = recall_score(self.y, y_pred, zero_division=0)
            f1 = f1_score(self.y, y_pred, zero_division=0)
            roc_auc = roc_auc_score(self.y, y_proba)
            cm = confusion_matrix(self.y, y_pred)

            print(f"  Accuracy:   {acc:.4f}")
            print(f"  Precision:  {prec:.4f}")
            print(f"  Recall:     {rec:.4f}")
            print(f"  F1-score:   {f1:.4f}")
            print(f"  ROC-AUC:    {roc_auc:.4f}")
            print(f"  混淆矩阵:")
            print(f"    TN={cm[0][0]}, FP={cm[0][1]}")
            print(f"    FN={cm[1][0]}, TP={cm[1][1]}")

            results[model_name] = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "roc_auc": round(roc_auc, 4),
                "confusion_matrix": {"TN": int(cm[0][0]), "FP": int(cm[0][1]),
                                     "FN": int(cm[1][0]), "TP": int(cm[1][1])},
                "y_pred": y_pred.tolist(),
                "y_proba": y_proba.tolist(),
            }

        self.results = results
        return results

    def load_lab2_results(self):
        """加载 lab2 的人类代码结果用于对比"""
        lab2_eval_json = os.path.join(config.LAB2_BASE_DIR, "data", "results", "evaluation.json")
        if os.path.exists(lab2_eval_json):
            with open(lab2_eval_json, "r", encoding="utf-8") as f:
                return json.load(f)

        lab2_report = os.path.join(config.LAB2_BASE_DIR, "data", "results", "evaluation_report.txt")
        if os.path.exists(lab2_report):
            lab2_results = {}
            with open(lab2_report, "r", encoding="utf-8") as f:
                content = f.read()
            return {"source": "evaluation_report.txt", "content": content}

        return {"source": "unknown", "note": "未找到 lab2 评估结果"}

    def save_results(self):
        """保存预测结果"""
        output = {
            "ai_code_results": self.results,
            "lab2_human_results": self.load_lab2_results(),
        }

        with open(config.STEP2_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {config.STEP2_RESULTS_PATH}")

    def plot_comparison(self):
        """绘制对比图表"""
        print("\n生成对比图表...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. 整体指标对比
        models = ["SVM", "Random Forest"]
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

        x = np.arange(len(metrics))
        width = 0.35
        ax = axes[0, 0]

        for i, model_name in enumerate(models):
            vals = [self.results[model_name][metric] for metric in metrics]
            offset = width * (i - 0.5)
            ax.bar(x + offset, vals, width, label=model_name,
                   color=["#ff6b6b", "#4dabf7"][i], edgecolor="black")
            for j, v in enumerate(vals):
                ax.text(x[j] + offset, v + 0.02, f"{v:.3f}", ha="center", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title("AI Code: SVM vs RF Performance", fontsize=13, fontweight="bold")
        ax.legend()

        # 2. SVM 混淆矩阵
        ax = axes[0, 1]
        svm_cm = self.results["SVM"]["confusion_matrix"]
        cm = np.array([[svm_cm["TN"], svm_cm["FP"]], [svm_cm["FN"], svm_cm["TP"]]])
        im = ax.imshow(cm, cmap="Reds")
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
        ax.set_title("SVM on AI Code - Confusion Matrix", fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax)

        # 3. RF 混淆矩阵
        ax = axes[1, 0]
        rf_cm = self.results["Random Forest"]["confusion_matrix"]
        cm = np.array([[rf_cm["TN"], rf_cm["FP"]], [rf_cm["FN"], rf_cm["TP"]]])
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
        ax.set_title("Random Forest on AI Code - Confusion Matrix", fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax)

        # 4. ROC 曲线
        ax = axes[1, 1]
        for model_name, color in [("SVM", "#ff6b6b"), ("Random Forest", "#4dabf7")]:
            y_proba = np.array(self.results[model_name]["y_proba"])
            fpr, tpr, _ = roc_curve(self.y, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{model_name} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve on AI Code", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(config.LAB5_FIGURES_DIR, "07_step2_ml_results.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {save_path}")

    def run_all(self):
        """执行完整流程"""
        print("=" * 60)
        print("步骤二: 传统机器学习模型 (SVM, Random Forest)")
        print("=" * 60)

        self.load_features()
        self.load_labels()
        self.load_models()
        self.predict_and_evaluate()
        self.save_results()
        self.plot_comparison()

        print("\n" + "=" * 60)
        print("步骤二完成!")
        print("=" * 60)
        return self.results

    def print_summary(self):
        """打印结果摘要"""
        print("\n" + "=" * 60)
        print("步骤二结果摘要")
        print("=" * 60)

        print(f"\n样本数量: {self.results['data_info']['total_samples']}")
        print(f"  Merged: {self.results['data_info']['merged_count']} "
              f"({self.results['data_info']['merged_ratio']:.1%})")
        print(f"  Not Merged: {self.results['data_info']['not_merged_count']}")

        print(f"\n{'模型':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}")
        print("-" * 60)
        for model_name in ["SVM", "Random Forest"]:
            r = self.results[model_name]
            print(f"{model_name:<20} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
                  f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['roc_auc']:>8.4f}")


def main():
    predictor = MLPredictor()
    results = predictor.run_all()
    predictor.print_summary()
    return results


if __name__ == "__main__":
    main()