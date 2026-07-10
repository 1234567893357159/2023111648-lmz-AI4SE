"""
步骤一：AI 生成代码数据筛选与统计分析
从 lab1 的 dataset.csv 中筛选 AI 生成代码的 PR，完成统计和可视化
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
sns.set_style("whitegrid")
sns.set_palette("husl")


class AIPRFilter:
    """AI 代码 PR 筛选器"""

    def __init__(self):
        self.df = None
        self.ai_df = None
        self.human_df = None
        self.stats = {}

    def load_dataset(self) -> pd.DataFrame:
        """加载 lab1 数据集"""
        print(f"加载 lab1 数据集: {config.LAB1_DATASET_PATH}")
        self.df = pd.read_csv(config.LAB1_DATASET_PATH)
        print(f"  总 PR 数: {len(self.df)}")
        print(f"  特征数: {len(self.df.columns)}")
        return self.df

    def filter_ai_prs(self) -> pd.DataFrame:
        """筛选 AI 生成代码的 PR"""
        if self.df is None:
            self.load_dataset()

        self.ai_df = self.df[self.df["has_ai_generated_code"] == 1].copy()
        self.human_df = self.df[self.df["has_ai_generated_code"] == 0].copy()

        print(f"\n筛选结果:")
        print(f"  AI 代码 PR: {len(self.ai_df)} ({len(self.ai_df)/len(self.df)*100:.1f}%)")
        print(f"  人类代码 PR: {len(self.human_df)} ({len(self.human_df)/len(self.df)*100:.1f}%)")

        return self.ai_df

    def save_filtered_data(self):
        """保存筛选结果"""
        self.ai_df.to_csv(config.AI_PRS_PATH, index=False)
        print(f"\nAI 代码 PR 已保存: {config.AI_PRS_PATH}")

        self.human_df.to_csv(config.HUMAN_PRS_PATH, index=False)
        print(f"人类代码 PR 已保存: {config.HUMAN_PRS_PATH}")

    def compute_statistics(self) -> dict:
        """计算统计分析指标"""
        print("\n" + "=" * 60)
        print("统计分析")
        print("=" * 60)

        stats = {}

        stats["total_prs"] = len(self.df)
        stats["ai_pr_count"] = len(self.ai_df)
        stats["human_pr_count"] = len(self.human_df)
        stats["ai_pr_ratio"] = round(len(self.ai_df) / len(self.df), 4)

        stats["ai_merge_rate"] = round(self.ai_df["is_merged"].mean(), 4)
        stats["human_merge_rate"] = round(self.human_df["is_merged"].mean(), 4)
        print(f"\nMerge Rate 对比:")
        print(f"  AI 代码: {stats['ai_merge_rate']:.2%}")
        print(f"  人类代码: {stats['human_merge_rate']:.2%}")
        print(f"  差异: {abs(stats['ai_merge_rate'] - stats['human_merge_rate']):.2%}")

        stats["per_repo"] = {}
        for owner, repo in config.TARGET_REPOS:
            key = f"{owner}/{repo}"
            total_repo = len(self.df[(self.df["repo_owner"] == owner) & (self.df["repo_name"] == repo)])
            ai_repo = len(self.ai_df[(self.ai_df["repo_owner"] == owner) & (self.ai_df["repo_name"] == repo)])
            stats["per_repo"][key] = {
                "total": total_repo,
                "ai_count": ai_repo,
                "ai_ratio": round(ai_repo / total_repo, 4) if total_repo > 0 else 0,
            }
        print(f"\n各仓库 AI PR 分布:")
        for repo, info in stats["per_repo"].items():
            print(f"  {repo}: {info['ai_count']}/{info['total']} ({info['ai_ratio']:.1%})")

        stats["ai_indicators"] = self.ai_df["ai_code_indicators"].value_counts().head(10).to_dict()
        print(f"\nAI 代码检测来源 Top 10:")
        for indicator, count in list(stats["ai_indicators"].items())[:10]:
            print(f"  {indicator}: {count}")

        stats["ai_language_dist"] = self.ai_df["primary_language"].value_counts().to_dict()
        stats["human_language_dist"] = self.human_df["primary_language"].value_counts().to_dict()
        print(f"\nAI 代码 PR 语言分布:")
        for lang, count in sorted(stats["ai_language_dist"].items(), key=lambda x: -x[1]):
            print(f"  {lang}: {count}")

        feature_cols = [
            ("files_changed", "修改文件数"),
            ("additions", "新增行数"),
            ("deletions", "删除行数"),
            ("pr_length", "PR 描述长度"),
            ("title_length", "标题长度"),
            ("reviewer_count", "Reviewer 数量"),
            ("total_comment_count", "总评论数"),
            ("label_count", "标签数"),
            ("modified_function_count", "修改函数数"),
        ]
        stats["feature_comparison"] = {}
        print(f"\n特征对比 (AI vs Human):")
        for col, name in feature_cols:
            if col in self.df.columns:
                ai_val = round(self.ai_df[col].mean(), 2)
                human_val = round(self.human_df[col].mean(), 2)
                stats["feature_comparison"][col] = {
                    "name": name,
                    "ai_mean": ai_val,
                    "human_mean": human_val,
                }
                print(f"  {name}: AI={ai_val} vs Human={human_val}")

        self.stats = stats
        return stats

    def save_statistics(self):
        """保存统计结果"""
        with open(config.AI_STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        print(f"\n统计结果已保存: {config.AI_STATS_PATH}")

    def generate_visualizations(self):
        """生成可视化图表"""
        print("\n" + "=" * 60)
        print("生成可视化图表")
        print("=" * 60)

        self._plot_ai_vs_human_count()
        self._plot_per_repo_distribution()
        self._plot_merge_rate_comparison()
        self._plot_feature_comparison()
        self._plot_ai_indicator_distribution()
        self._plot_language_distribution()

        print("\n所有图表生成完成！")

    def _plot_ai_vs_human_count(self):
        """AI vs Human PR 数量饼图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        labels = ["Human Code", "AI Generated"]
        sizes = [len(self.human_df), len(self.ai_df)]
        colors = ["#4dabf7", "#ff6b6b"]
        explode = (0, 0.05)

        axes[0].pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors,
                    explode=explode, startangle=90, shadow=True)
        axes[0].set_title("AI vs Human PR Ratio", fontsize=14, fontweight="bold")

        axes[1].bar(labels, sizes, color=colors, edgecolor="black", linewidth=1.2)
        axes[1].set_title("AI vs Human PR Count", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("PR Count")
        for i, v in enumerate(sizes):
            axes[1].text(i, v + 10, str(v), ha="center", fontsize=12, fontweight="bold")

        plt.tight_layout()
        path = os.path.join(config.LAB5_FIGURES_DIR, "01_ai_vs_human_count.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {path}")

    def _plot_per_repo_distribution(self):
        """各仓库 AI PR 分布"""
        repos = []
        ai_counts = []
        human_counts = []
        for owner, repo in config.TARGET_REPOS:
            ai_count = len(self.ai_df[(self.ai_df["repo_owner"] == owner) & (self.ai_df["repo_name"] == repo)])
            human_count = len(self.human_df[(self.human_df["repo_owner"] == owner) & (self.human_df["repo_name"] == repo)])
            repos.append(repo)
            ai_counts.append(ai_count)
            human_counts.append(human_count)

        x = np.arange(len(repos))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, human_counts, width, label="Human Code", color="#4dabf7", edgecolor="black")
        bars2 = ax.bar(x + width/2, ai_counts, width, label="AI Generated", color="#ff6b6b", edgecolor="black")

        ax.set_xlabel("Repository")
        ax.set_ylabel("PR Count")
        ax.set_title("AI vs Human PR Distribution per Repository", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(repos)
        ax.legend()

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(int(bar.get_height())), ha="center", fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(int(bar.get_height())), ha="center", fontsize=9)

        plt.tight_layout()
        path = os.path.join(config.LAB5_FIGURES_DIR, "02_per_repo_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {path}")

    def _plot_merge_rate_comparison(self):
        """Merge Rate 对比"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        categories = ["Human Code", "AI Generated"]
        merge_rates = [self.human_df["is_merged"].mean(), self.ai_df["is_merged"].mean()]
        colors = ["#4dabf7", "#ff6b6b"]
        axes[0].bar(categories, merge_rates, color=colors, edgecolor="black", linewidth=1.2)
        axes[0].set_title("Overall Merge Rate Comparison", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Merge Rate")
        axes[0].set_ylim(0, 1)
        for i, v in enumerate(merge_rates):
            axes[0].text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=12, fontweight="bold")

        repo_names = []
        ai_rates = []
        human_rates = []
        for owner, repo in config.TARGET_REPOS:
            repo_names.append(repo)
            repo_ai = self.ai_df[(self.ai_df["repo_owner"] == owner) & (self.ai_df["repo_name"] == repo)]
            repo_human = self.human_df[(self.human_df["repo_owner"] == owner) & (self.human_df["repo_name"] == repo)]
            ai_rates.append(repo_ai["is_merged"].mean() if len(repo_ai) > 0 else 0)
            human_rates.append(repo_human["is_merged"].mean() if len(repo_human) > 0 else 0)

        x = np.arange(len(repo_names))
        width = 0.35
        axes[1].bar(x - width/2, human_rates, width, label="Human Code", color="#4dabf7", edgecolor="black")
        axes[1].bar(x + width/2, ai_rates, width, label="AI Generated", color="#ff6b6b", edgecolor="black")
        axes[1].set_title("Merge Rate per Repository", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Merge Rate")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(repo_names)
        axes[1].set_ylim(0, 1)
        axes[1].legend()

        plt.tight_layout()
        path = os.path.join(config.LAB5_FIGURES_DIR, "03_merge_rate_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {path}")

    def _plot_feature_comparison(self):
        """特征对比图"""
        feature_cols = [
            ("files_changed", "Files Changed"),
            ("additions", "Additions"),
            ("deletions", "Deletions"),
            ("pr_length", "PR Length"),
            ("title_length", "Title Length"),
            ("reviewer_count", "Reviewers"),
            ("total_comment_count", "Comments"),
            ("label_count", "Labels"),
            ("modified_function_count", "Modified Funcs"),
        ]

        names = []
        ai_vals = []
        human_vals = []
        for col, name in feature_cols:
            if col in self.df.columns:
                names.append(name)
                ai_vals.append(self.ai_df[col].mean())
                human_vals.append(self.human_df[col].mean())

        x = np.arange(len(names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(14, 6))
        bars1 = ax.bar(x - width/2, human_vals, width, label="Human Code", color="#4dabf7", edgecolor="black")
        bars2 = ax.bar(x + width/2, ai_vals, width, label="AI Generated", color="#ff6b6b", edgecolor="black")

        ax.set_xlabel("Feature")
        ax.set_ylabel("Mean Value")
        ax.set_title("Feature Comparison: AI vs Human", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.legend()

        plt.tight_layout()
        path = os.path.join(config.LAB5_FIGURES_DIR, "04_feature_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {path}")

    def _plot_ai_indicator_distribution(self):
        """AI 代码检测来源分布"""
        indicator_counts = self.ai_df["ai_code_indicators"].value_counts().head(10)

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(range(len(indicator_counts)), indicator_counts.values, color="#ff6b6b", edgecolor="black")
        ax.set_yticks(range(len(indicator_counts)))
        ax.set_yticklabels(indicator_counts.index, fontsize=8)
        ax.set_xlabel("Count")
        ax.set_title("AI Code Detection Indicator Distribution (Top 10)", fontsize=14, fontweight="bold")
        ax.invert_yaxis()

        for bar, v in zip(bars, indicator_counts.values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    str(v), va="center", fontsize=9)

        plt.tight_layout()
        path = os.path.join(config.LAB5_FIGURES_DIR, "05_ai_indicator_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {path}")

    def _plot_language_distribution(self):
        """语言分布对比"""
        ai_lang = self.ai_df["primary_language"].value_counts()
        human_lang = self.human_df["primary_language"].value_counts()

        all_langs = sorted(set(ai_lang.index) | set(human_lang.index),
                          key=lambda x: ai_lang.get(x, 0) + human_lang.get(x, 0), reverse=True)
        all_langs = all_langs[:10]

        x = np.arange(len(all_langs))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, [human_lang.get(l, 0) for l in all_langs], width,
               label="Human Code", color="#4dabf7", edgecolor="black")
        ax.bar(x + width/2, [ai_lang.get(l, 0) for l in all_langs], width,
               label="AI Generated", color="#ff6b6b", edgecolor="black")

        ax.set_xlabel("Programming Language")
        ax.set_ylabel("PR Count")
        ax.set_title("Language Distribution: AI vs Human", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(all_langs, rotation=30, ha="right")
        ax.legend()

        plt.tight_layout()
        path = os.path.join(config.LAB5_FIGURES_DIR, "06_language_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {path}")

    def run_all(self):
        """执行完整流程"""
        print("=" * 60)
        print("步骤一：AI 生成代码数据筛选与统计分析")
        print("=" * 60)

        self.load_dataset()
        self.filter_ai_prs()
        self.save_filtered_data()
        self.compute_statistics()
        self.save_statistics()
        self.generate_visualizations()

        print("\n" + "=" * 60)
        print("步骤一完成！")
        print("=" * 60)
        return self.ai_df, self.human_df, self.stats


def main():
    filter_obj = AIPRFilter()
    ai_df, human_df, stats = filter_obj.run_all()
    return ai_df, human_df, stats


if __name__ == "__main__":
    main()