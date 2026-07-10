"""
步骤三 Task 3: CodeT5 Review Comment Generation 在 AI 代码上的推理评估
只加载 lab3 微调好的模型，不重新训练
"""

import json
import os
import re
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import config
importlib.reload(config)

sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

AUTO_GENERATED_PATTERNS = [
    r"(^|/)package-lock\.json$", r"(^|/)yarn\.lock$", r"(^|/)pnpm-lock\.yaml$",
    r"(^|/)go\.sum$", r"(^|/)Cargo\.lock$", r"(^|/)Gemfile\.lock$",
    r"(^|/)poetry\.lock$", r"(^|/)Pipfile\.lock$",
    r"\.pb\.go$", r"\.pb\.cc$", r"\.pb\.h$", r"\.pb\.[a-z]+$",
    r"\.generated\.\w+$", r"\.auto\.\w+$", r"\.min\.js$", r"\.min\.css$",
    r"\.pyc$", r"\.pyo$", r"(^|/)vendor/", r"(^|/)node_modules/",
    r"(^|/)__pycache__/", r"\.patch$", r"\.diff$", r"\.lock$",
]


def is_auto_generated(filename):
    for pattern in AUTO_GENERATED_PATTERNS:
        if re.search(pattern, filename):
            return True
    return False


def compress_patch(patch, max_chars=1200):
    lines = patch.split("\n")
    compressed = []
    context_count = 0
    for line in lines:
        if line.startswith("@@"):
            if context_count > 0:
                compressed.append(f"... ({context_count} context lines omitted)")
                context_count = 0
            compressed.append(line)
        elif line.startswith("+") or line.startswith("-"):
            if context_count > 0:
                compressed.append(f"... ({context_count} context lines omitted)")
                context_count = 0
            compressed.append(line)
        elif line.startswith(" ") or line == "":
            context_count += 1
        else:
            if context_count > 0:
                compressed.append(f"... ({context_count} context lines omitted)")
                context_count = 0
            compressed.append(line)
    if context_count > 0:
        compressed.append(f"... ({context_count} context lines omitted)")
    result = "\n".join(compressed)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... (truncated)"
    return result


class CodeT5Evaluator:
    """CodeT5 推理评估器：加载 lab3 微调好的模型 + 对 AI 代码生成评论"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        self.tokenizer = None
        self.model = None
        self.results = {}
        self.samples = []

    def load_model(self):
        print("加载 lab3 CodeT5 模型...")
        model_dir = config.LAB3_CODET5_MODEL_DIR
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"  模型路径: {model_dir}")

    def _build_input_for_file(self, pr, file_info):
        parts = []
        title = pr.get("title", "")
        if title:
            parts.append(f"title: {title}")
        body = pr.get("body", "")
        if body:
            parts.append(f"body: {body}")
        commits = pr.get("commits", [])
        if commits:
            commit_msgs = [c.get("message", "")[:200] for c in commits]
            parts.append(f"commits: {' | '.join(commit_msgs)}")
        filename = file_info.get("filename", "")
        status = file_info.get("status", "")
        additions = file_info.get("additions", 0)
        deletions = file_info.get("deletions", 0)
        parts.append(f"file: {filename} (status: {status}, +{additions} -{deletions})")
        patch = file_info.get("patch", "")
        if patch:
            compressed = compress_patch(patch)
            parts.append(f"patch:\n{compressed}")
        return "\n".join(parts)

    def _get_file_comments(self, pr, filename):
        review_comments = pr.get("review_comments", [])
        file_comments = []
        for rc in review_comments:
            rc_path = rc.get("path", "")
            if rc_path == filename:
                body = rc.get("body", "").strip()
                if body and len(body) > 3:
                    file_comments.append(body)
        return file_comments

    def prepare_data(self):
        print("准备 AI PR 代码审查数据...")
        import pandas as pd
        ai_df = pd.read_csv(config.AI_PRS_PATH)
        ai_pr_ids = set(ai_df["pr_id"].tolist())

        raw_dir = config.LAB1_RAW_DIR
        pr_cache = {}
        for fname in sorted(os.listdir(raw_dir)):
            if not fname.endswith("_pulls.json"):
                continue
            with open(os.path.join(raw_dir, fname), "r", encoding="utf-8") as f:
                prs = json.load(f)
            for pr in prs:
                pr_cache[pr["pr_id"]] = pr

        samples = []
        for pr_id in ai_pr_ids:
            pr = pr_cache.get(pr_id)
            if pr is None:
                continue
            files = pr.get("files", [])
            for file_info in files:
                filename = file_info.get("filename", "")
                patch = file_info.get("patch", "")
                if not patch:
                    continue
                if is_auto_generated(filename):
                    continue
                file_comments = self._get_file_comments(pr, filename)
                if not file_comments:
                    continue
                input_text = self._build_input_for_file(pr, file_info)
                target_text = " | ".join(file_comments)
                samples.append({
                    "pr_id": pr_id,
                    "filename": filename,
                    "input": input_text,
                    "target": target_text,
                })

        self.samples = samples
        print(f"  有效样本数: {len(samples)}")
        return samples

    def generate_and_evaluate(self):
        print("生成审查意见并评估...")
        predictions = []
        targets = []

        batch_size = 4
        for i in tqdm(range(0, len(self.samples), batch_size), desc="CodeT5 生成"):
            batch = self.samples[i:i + batch_size]
            inputs = self.tokenizer(
                [s["input"] for s in batch],
                truncation=True, padding=True, max_length=config.CODET5_MAX_INPUT_LENGTH,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=config.CODET5_MAX_TARGET_LENGTH,
                    num_beams=4,
                    early_stopping=True,
                )

            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend([d.strip() for d in decoded])
            targets.extend([s["target"] for s in batch])

        smooth = SmoothingFunction().method1
        bleu_scores = []
        for pred, target in zip(predictions, targets):
            if not pred or not target:
                bleu_scores.append(0.0)
            else:
                bleu = sentence_bleu(
                    [target.split()], pred.split(),
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smooth,
                )
                bleu_scores.append(bleu)

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores = []
        for pred, target in zip(predictions, targets):
            if not pred or not target:
                rouge_scores.append(0.0)
            else:
                scores = scorer.score(target, pred)
                rouge_scores.append(scores["rougeL"].fmeasure)

        self.results = {
            "n_samples": len(predictions),
            "bleu": float(np.mean(bleu_scores)),
            "rougeL": float(np.mean(rouge_scores)),
        }

        print(f"\n  样本数: {len(predictions)}")
        print(f"  BLEU-4:  {self.results['bleu']:.4f}")
        print(f"  ROUGE-L: {self.results['rougeL']:.4f}")

        return predictions, targets, bleu_scores, rouge_scores

    def save_results(self, predictions, targets):
        random_indices = np.random.choice(
            min(len(predictions), len(targets)),
            size=min(10, len(predictions)), replace=False,
        )
        examples = []
        for i in random_indices:
            examples.append({
                "pr_id": int(self.samples[i]["pr_id"]),
                "filename": self.samples[i]["filename"],
                "input_preview": self.samples[i]["input"][:300],
                "target": targets[i],
                "prediction": predictions[i],
            })

        output = {
            "n_samples": len(predictions),
            "metrics": self.results,
            "examples": examples,
        }
        lab3_res_path = os.path.join(config.LAB3_BASE_DIR, "data", "results", "classification_results.json")
        if os.path.exists(lab3_res_path):
            with open(lab3_res_path, "r", encoding="utf-8") as f:
                lab3_res = json.load(f)
            if "CodeT5" in lab3_res:
                output["human_baseline"] = lab3_res["CodeT5"]
            else:
                output["human_baseline"] = {}

        with open(config.STEP3_TASK3_CODET5_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {config.STEP3_TASK3_CODET5_PATH}")

    def plot_comparison(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        metrics = ["bleu", "rougeL"]
        labels = ["BLEU-4", "ROUGE-L"]
        values = [self.results.get(m, 0) for m in metrics]
        x = np.arange(len(metrics))
        bars = ax.bar(x, values, 0.4, color=["#ff6b6b", "#4dabf7"], edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("CodeT5 Review Generation: AI Code Performance", fontsize=14, fontweight="bold")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.4f}", ha="center", fontsize=12, fontweight="bold")

        plt.tight_layout()
        fig_path = os.path.join(config.LAB5_FIGURES_DIR, "10_step3_codet5_comparison.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"图表已保存: {fig_path}")

    def run_all(self):
        print("=" * 60)
        print("步骤三 Task 3: CodeT5 Review Comment Generation 推理评估")
        print("=" * 60)
        self.load_model()
        self.prepare_data()
        predictions, targets, _, _ = self.generate_and_evaluate()
        self.save_results(predictions, targets)
        self.plot_comparison()
        print("\n" + "=" * 60)
        print("Task 3 完成!")
        print("=" * 60)
        return self.results

    def print_summary(self):
        print("\n" + "=" * 60)
        print("CodeT5 结果摘要")
        print("=" * 60)
        print(f"\n样本数: {self.results.get('n_samples', 0)}")
        for k, v in self.results.items():
            if k != "n_samples":
                print(f"  {k:>10}: {v:.4f}")