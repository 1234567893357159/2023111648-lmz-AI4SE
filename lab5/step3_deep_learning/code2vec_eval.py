"""
步骤三 Task 1: Code2Vec + SVM/RF/MLP 在 AI 代码上的推理评估
只加载 lab3 训练好的分类器，不重新训练
"""

import json
import os
import sys
import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "lab3", "task1_code2vec"))

import importlib
import config
importlib.reload(config)

from ast_path_extractor import ASTPathExtractor
from vocab_builder import VocabBuilder
from code2vec_model import create_model

sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class Code2VecEvaluator:
    """Code2Vec 推理评估器：加载 lab3 分类器 + 对 AI 代码做预测"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = ASTPathExtractor()
        self.vocab = None
        self.model = None
        self.results = {}
        self.y_true = None
        self.predictions = {}
        self.probabilities = {}

    def load_vocab(self):
        print("加载 lab3 词汇表...")
        self.vocab = VocabBuilder()
        self.vocab.load()
        print(f"  Token 词汇表: {len(self.vocab.token_to_id)}")
        print(f"  Path 词汇表: {len(self.vocab.path_to_id)}")

    def create_code2vec(self):
        print("创建 Code2Vec 模型（与 lab3 相同架构）...")
        self.model = create_model(self.vocab)
        self.model = self.model.to(self.device)
        self.model.eval()
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  参数量: {total_params:,}")

        if os.path.exists(config.LAB3_CODE2VEC_MODEL_PATH):
            self.model.load_state_dict(torch.load(config.LAB3_CODE2VEC_MODEL_PATH, map_location=self.device))
            print(f"  已加载 lab3 模型权重: {config.LAB3_CODE2VEC_MODEL_PATH}")
        else:
            print(f"  警告: 未找到 lab3 模型权重 ({config.LAB3_CODE2VEC_MODEL_PATH})，使用随机权重")

    def _load_ai_pr_list(self):
        pulls_dir = config.AI_PULLS_DIR
        pr_list = []
        for fname in os.listdir(pulls_dir):
            if fname.endswith("_pulls.json"):
                repo_key = fname.replace("_pulls.json", "")
                with open(os.path.join(pulls_dir, fname), "r", encoding="utf-8") as f:
                    prs = json.load(f)
                for pr in prs:
                    pr["_repo_key"] = repo_key
                pr_list.extend(prs)
        return pr_list

    def _paths_to_ids(self, paths, max_paths):
        if not paths:
            return torch.zeros(3, 1, dtype=torch.long)
        n = min(len(paths), max_paths)
        starts = torch.zeros(n, dtype=torch.long)
        path_ids = torch.zeros(n, dtype=torch.long)
        ends = torch.zeros(n, dtype=torch.long)
        for i in range(n):
            start_token, path_str, end_token = paths[i]
            starts[i] = self.vocab.token_to_idx(start_token)
            path_ids[i] = self.vocab.path_to_idx(path_str)
            ends[i] = self.vocab.token_to_idx(end_token)
        return torch.stack([starts, path_ids, ends], dim=0)

    def _extract_paths_for_pr(self, pr):
        pr_id = pr["pr_id"]
        repo_key = pr.get("_repo_key", "")
        code_dir = os.path.join(config.AI_CODE_DIR, repo_key, str(pr_id))

        before_dir = os.path.join(code_dir, "before")
        after_dir = os.path.join(code_dir, "after")

        before_paths = self.extractor.extract_paths_from_dir(before_dir) if os.path.isdir(before_dir) else []
        after_paths = self.extractor.extract_paths_from_dir(after_dir) if os.path.isdir(after_dir) else []
        return before_paths, after_paths

    def vectorize_all_prs(self, force_reevaluate=False):
        if not force_reevaluate and os.path.exists(config.STEP3_CACHE_VECTORS):
            print(f"加载缓存向量: {config.STEP3_CACHE_VECTORS}")
            data = torch.load(config.STEP3_CACHE_VECTORS, map_location="cpu")
            vectors = data["vectors"].numpy()
            if len(data["pr_ids"]) == 0:
                print("  缓存为空，重新生成...")
            else:
                return data["pr_ids"], data["labels"], vectors

        pr_list = self._load_ai_pr_list()
        print(f"向量化 {len(pr_list)} 个 AI PR...")

        pr_ids = []
        labels = []
        vectors = []

        self.model.eval()
        with torch.no_grad():
            for pr in tqdm(pr_list, desc="向量化"):
                before_paths, after_paths = self._extract_paths_for_pr(pr)
                if not before_paths and not after_paths:
                    continue

                before_ids = self._paths_to_ids(before_paths, config.MAX_PATHS_PER_FILE)
                after_ids = self._paths_to_ids(after_paths, config.MAX_PATHS_PER_FILE)

                before_ids_batch = before_ids.unsqueeze(0).to(self.device)
                after_ids_batch = after_ids.unsqueeze(0).to(self.device)

                before_vec = self.model(
                    before_ids_batch[:, 0, :], before_ids_batch[:, 1, :], before_ids_batch[:, 2, :]
                ).cpu()
                after_vec = self.model(
                    after_ids_batch[:, 0, :], after_ids_batch[:, 1, :], after_ids_batch[:, 2, :]
                ).cpu()

                pr_vec = torch.cat([before_vec, after_vec], dim=1).squeeze(0).numpy()

                pr_ids.append(pr["pr_id"])
                labels.append(1 if pr.get("merged", False) else 0)
                vectors.append(pr_vec)

        if len(vectors) == 0:
            raise RuntimeError(
                "向量化失败: 没有 PR 成功提取 AST 路径。\n"
                "请确认 AI 代码已下载到: " + config.AI_CODE_DIR + "\n"
                "请先运行步骤二的数据准备 (step2_traditional_ml/data_prepare.py) 或设置 force_reevaluate=True 并确保代码已存在"
            )

        pr_ids = np.array(pr_ids)
        labels = np.array(labels)
        vectors = np.array(vectors)

        torch.save(
            {"pr_ids": pr_ids, "labels": labels, "vectors": torch.tensor(vectors)},
            config.STEP3_CACHE_VECTORS,
        )
        print(f"  向量缓存已保存: {len(pr_ids)} 个 PR, 维度 {vectors.shape[1]}")
        return pr_ids, labels, vectors

    def load_classifiers(self):
        print("加载 lab3 训练好的分类器...")
        self.clf_svm = joblib.load(os.path.join(config.LAB3_MODELS_DIR, "svm.pkl"))
        print(f"  SVM: {config.LAB3_MODELS_DIR}/svm.pkl")
        self.clf_rf = joblib.load(os.path.join(config.LAB3_MODELS_DIR, "random_forest.pkl"))
        print(f"  RF: {config.LAB3_MODELS_DIR}/random_forest.pkl")
        mlp_data = joblib.load(os.path.join(config.LAB3_MODELS_DIR, "mlp.pkl"))
        self.clf_mlp = mlp_data["model"] if isinstance(mlp_data, dict) else mlp_data
        print(f"  MLP: {config.LAB3_MODELS_DIR}/mlp.pkl")

    def predict_and_evaluate(self, vectors, labels):
        vectors = np.array(vectors)
        if vectors.ndim == 1:
            vectors = vectors.reshape(-1, 1)
        labels = np.array(labels)

        self.y_true = labels
        print(f"\n样本数: {len(labels)}, merged={labels.sum()}, not_merged={(1-labels).sum()}")

        if len(vectors) == 0:
            raise RuntimeError(
                "没有可用的向量数据！请先运行 step2 下载 AI 代码并生成 AST。\n"
                f"如果已运行过 step2，请删除缓存文件后重试:\n"
                f"  {config.STEP3_CACHE_VECTORS}"
            )

        for name, clf in [("SVM", self.clf_svm), ("Random Forest", self.clf_rf), ("MLP", self.clf_mlp)]:
            print(f"\n{'='*50}")
            print(f"评估 {name}")
            print(f"{'='*50}")

            y_pred = clf.predict(vectors)
            self.predictions[name] = y_pred

            if hasattr(clf, "predict_proba"):
                y_proba = clf.predict_proba(vectors)[:, 1]
                self.probabilities[name] = y_proba
            elif hasattr(clf, "decision_function"):
                y_proba = clf.decision_function(vectors)
                y_proba = 1 / (1 + np.exp(-y_proba))
                self.probabilities[name] = y_proba

            self.results[name] = {
                "accuracy": accuracy_score(labels, y_pred),
                "precision": precision_score(labels, y_pred, zero_division=0),
                "recall": recall_score(labels, y_pred, zero_division=0),
                "f1": f1_score(labels, y_pred, zero_division=0),
            }
            if name in self.probabilities:
                self.results[name]["roc_auc"] = roc_auc_score(labels, self.probabilities[name])

            for k, v in self.results[name].items():
                print(f"  {k:>10}: {v:.4f}")

            cm = confusion_matrix(labels, y_pred)
            print(f"  混淆矩阵: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    def save_results(self):
        output = {
            "n_samples": int(len(self.y_true)),
            "n_merged": int(self.y_true.sum()),
            "n_not_merged": int((1 - self.y_true).sum()),
            "models": self.results,
            "human_baseline": {
                "SVM": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.0},
                "Random Forest": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.0},
                "MLP": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.0},
            },
        }
        lab3_res_path = os.path.join(config.LAB3_BASE_DIR, "data", "results", "classification_results.json")
        if os.path.exists(lab3_res_path):
            with open(lab3_res_path, "r", encoding="utf-8") as f:
                lab3_res = json.load(f)
            for model_name in ["SVM", "Random Forest", "MLP"]:
                if model_name in lab3_res:
                    output["human_baseline"][model_name] = lab3_res[model_name]

        with open(config.STEP3_TASK1_CODE2VEC_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {config.STEP3_TASK1_CODE2VEC_PATH}")

    def plot_comparison(self):
        models = list(self.results.keys())
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("Code2Vec: Human vs AI Code Performance", fontsize=16, fontweight="bold")

        x = np.arange(len(metrics))
        width = 0.25

        ax = axes[0, 0]
        for i, model_name in enumerate(models):
            vals = [self.results[model_name].get(m, 0) for m in metrics]
            offset = width * (i - 1)
            bars = ax.bar(x + offset, vals, width, label=f"AI {model_name}",
                          color=["#ff6b6b", "#4dabf7", "#51cf66"][i], edgecolor="black")
            for j, v in enumerate(vals):
                ax.text(x[j] + offset, v + 0.02, f"{v:.3f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title("AI Code: SVM vs RF vs MLP", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        model_name = "SVM"
        if model_name in self.results:
            ai_vals = [self.results[model_name].get(m, 0) for m in metrics]
            ax.bar(x - width/2, ai_vals, width, label="AI Code", color="#ff6b6b", edgecolor="black")
            for j, v in enumerate(ai_vals):
                ax.text(x[j] - width/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"SVM: AI Code Performance", fontsize=13, fontweight="bold")

        ax = axes[1, 0]
        model_name = "Random Forest"
        if model_name in self.results:
            ai_vals = [self.results[model_name].get(m, 0) for m in metrics]
            ax.bar(x - width/2, ai_vals, width, label="AI Code", color="#4dabf7", edgecolor="black")
            for j, v in enumerate(ai_vals):
                ax.text(x[j] - width/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"Random Forest: AI Code Performance", fontsize=13, fontweight="bold")

        ax = axes[1, 1]
        if self.probabilities:
            for name, y_proba in self.probabilities.items():
                fpr, tpr, _ = roc_curve(self.y_true, y_proba)
                auc_score = roc_auc_score(self.y_true, y_proba)
                ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_score:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)

        plt.tight_layout()
        fig_path = os.path.join(config.LAB5_FIGURES_DIR, "08_step3_code2vec_comparison.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"图表已保存: {fig_path}")

    def run_all(self, force_reevaluate=False):
        print("=" * 60)
        print("步骤三 Task 1: Code2Vec + SVM/RF/MLP 推理评估")
        print("=" * 60)
        self.load_vocab()
        self.create_code2vec()
        pr_ids, labels, vectors = self.vectorize_all_prs(force_reevaluate)
        self.load_classifiers()
        self.predict_and_evaluate(vectors, labels)
        self.save_results()
        self.plot_comparison()
        print("\n" + "=" * 60)
        print("Task 1 完成!")
        print("=" * 60)
        return self.results

    def print_summary(self):
        print("\n" + "=" * 60)
        print("Code2Vec 结果摘要")
        print("=" * 60)
        print(f"\n样本数: {len(self.y_true)}")
        print(f"  Merged: {self.y_true.sum()} ({self.y_true.sum()/len(self.y_true):.1%})")
        print()
        for name, metrics in self.results.items():
            print(f"  {name}:")
            for k, v in metrics.items():
                print(f"    {k:>10}: {v:.4f}")