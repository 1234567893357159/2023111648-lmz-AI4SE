"""
统计分析 & 可视化模块
完成实验要求的各项统计分析，并生成可视化图表
"""

import matplotlib
matplotlib.use('Agg')  # must be before plt import
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

# ====== 其他库导入 ======
import os
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.font_manager as fm
import seaborn as sns

import config

plt.rcParams['figure.dpi'] = 100
# 设置 Seaborn 样式
sns.set_style("whitegrid")
sns.set_palette("husl")


class DataAnalyzer:
    """数据分析器"""

    def __init__(self, df: pd.DataFrame):
        """
        初始化
        Args:
            df: 特征 DataFrame
        """
        self.df = df
        self.figures_dir = config.FIGURES_DIR
        os.makedirs(self.figures_dir, exist_ok=True)

    def print_data_overview(self) -> None:
        """打印数据概览"""
        print("=" * 60)
        print("Dataset Overview")
        print("=" * 60)
        print(f"Total PRs: {len(self.df)}")
        print(f"Features: {len(self.df.columns)}")
        print(f"Columns: {list(self.df.columns)}")
        print()

        # 按仓库分组统计
        print("-" * 40)
        print("PRs per Repository:")
        print("-" * 40)
        repo_counts = self.df.groupby(['repo_owner', 'repo_name']).size()
        for (owner, repo), count in repo_counts.items():
            merged = len(self.df[(self.df['repo_owner'] == owner) &
                                 (self.df['repo_name'] == repo) &
                                 (self.df['is_merged'] == 1)])
            print(f"  {owner}/{repo}: {count} PRs (merged: {merged})")
        print()

        # 缺失值检查
        print("-" * 40)
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            print("Missing values:")
            print(missing)
        else:
            print("No missing values")
        print()

    def analyze_merge_vs_non_merge(self) -> None:
        """
        分析 1: Merge 与 Non-Merge 数量统计
        生成柱状图 + 饼图
        """
        print("=" * 60)
        print("Analysis: Merge vs Non-Merge Count")
        print("=" * 60)

        merge_counts = self.df['is_merged'].value_counts()
        merge_labels = ['Non-Merge (0)', 'Merge (1)']
        print(f"  Merge: {merge_counts.get(1, 0)}")
        print(f"  Non-Merge: {merge_counts.get(0, 0)}")
        print(f"  Merge Rate: {merge_counts.get(1, 0) / len(self.df) * 100:.2f}%")
        print()

        # 柱状图
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 柱状图
        axes[0].bar(merge_labels, [merge_counts.get(0, 0), merge_counts.get(1, 0)],
                    color=['#ff6b6b', '#51cf66'], edgecolor='black', linewidth=1.2)
        axes[0].set_title('Merge vs Non-Merge Count', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('PR Count')
        for i, v in enumerate([merge_counts.get(0, 0), merge_counts.get(1, 0)]):
            axes[0].text(i, v + 5, str(v), ha='center', fontsize=12)

        # 饼图
        colors = ['#ff6b6b', '#51cf66']
        explode = (0, 0.05)
        axes[1].pie(
            [merge_counts.get(0, 0), merge_counts.get(1, 0)],
            labels=merge_labels,
            autopct='%1.1f%%',
            colors=colors,
            explode=explode,
            startangle=90,
            shadow=True
        )
        axes[1].set_title('Merge vs Non-Merge Ratio', fontsize=14, fontweight='bold')

        plt.tight_layout()
        file_path = os.path.join(self.figures_dir, '01_merge_vs_non_merge.png')
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"图表已保存: {file_path}\n")

    def analyze_comment_distribution(self) -> None:
        """
        分析 2: Review Comment 数量分布
        生成直方图
        """
        print("=" * 60)
        print("分析: Review Comment 数量分布")
        print("=" * 60)

        comment_col = 'total_comment_count'
        if comment_col not in self.df.columns:
            comment_col = 'review_comment_count'

        comments = self.df[comment_col]
        print(f"  mean: {comments.mean():.2f}")
        print(f"  median: {comments.median():.2f}")
        print(f"  min: {comments.min()}")
        print(f"  max: {comments.max()}")
        print(f"  std: {comments.std():.2f}")
        print()

        # 直方图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 全量分布
        axes[0].hist(comments, bins=50, color='#4dabf7', edgecolor='black',
                     alpha=0.8, linewidth=0.5)
        axes[0].axvline(comments.mean(), color='red', linestyle='--',
                        linewidth=2, label=f"mean={comments.mean():.1f}")
        axes[0].axvline(comments.median(), color='green', linestyle='--',
                        linewidth=2, label=f"median={comments.median():.1f}")
        axes[0].set_title('Review Comment Distribution (Full)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Comment Count')
        axes[0].set_ylabel('PR Count')
        axes[0].legend()

        # 截断分布（去掉尾部极端值）
        upper_limit = comments.quantile(0.95)
        comments_truncated = comments[comments <= upper_limit]
        axes[1].hist(comments_truncated, bins=40, color='#4dabf7', edgecolor='black',
                     alpha=0.8, linewidth=0.5)
        axes[1].axvline(comments_truncated.mean(), color='red', linestyle='--',
                        linewidth=2, label=f"mean={comments_truncated.mean():.1f}")
        axes[1].axvline(comments_truncated.median(), color='green', linestyle='--',
                        linewidth=2, label=f"median={comments_truncated.median():.1f}")
        axes[1].set_title(f'Review Comment Distribution (P5-P95)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Comment Count')
        axes[1].set_ylabel('PR Count')
        axes[1].legend()

        plt.tight_layout()
        file_path = os.path.join(self.figures_dir, '02_comment_distribution.png')
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"图表已保存: {file_path}\n")

    def analyze_label_distribution(self, top_n: int = 15) -> None:
        """
        分析 3: Label 分布
        生成柱状图（Top N）
        """
        print("=" * 60)
        print("分析: Label 分布")
        print("=" * 60)

        # 提取所有 Label
        all_labels = []
        for labels_str in self.df['labels']:
            if labels_str and isinstance(labels_str, str):
                for label in labels_str.split(','):
                    label = label.strip()
                    if label:
                        all_labels.append(label)

        if not all_labels:
            print("  无 Label 数据\n")
            return

        label_series = pd.Series(all_labels)
        label_counts = label_series.value_counts().head(top_n)

        print(f"  不同 Label 总数: {label_series.nunique()}")
        print(f"  Top {top_n} Labels:")
        for label, count in label_counts.items():
            print(f"    {label}: {count}")
        print()

        # 柱状图
        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.barh(range(len(label_counts)), label_counts.values, color='#845ef7',
                       edgecolor='black', linewidth=0.8)
        ax.set_yticks(range(len(label_counts)))
        ax.set_yticklabels(label_counts.index)
        ax.set_xlabel('Occurrences')
        ax.set_title(f'Label Distribution Top {top_n}', fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        for bar, v in zip(bars, label_counts.values):
            ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2, str(v),
                    va='center', fontsize=9)

        plt.tight_layout()
        file_path = os.path.join(self.figures_dir, '03_label_distribution.png')
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"图表已保存: {file_path}\n")

    def analyze_reviewer_count_distribution(self) -> None:
        """
        分析 4: Reviewer 数量分布
        生成直方图
        """
        print("=" * 60)
        print("分析: Reviewer 数量分布")
        print("=" * 60)

        if 'reviewer_count' not in self.df.columns:
            print("  缺少 reviewer_count 列\n")
            return

        reviewer_counts = self.df['reviewer_count']
        print(f"  mean: {reviewer_counts.mean():.2f}")
        print(f"  median: {reviewer_counts.median():.2f}")
        print(f"  min: {reviewer_counts.min()}")
        print(f"  max: {reviewer_counts.max()}")
        print(f"  无 Review 的 PR 数量: {(reviewer_counts == 0).sum()}")
        print()

        # 直方图
        fig, ax = plt.subplots(figsize=(10, 5))

        max_reviewer = min(reviewer_counts.max(), 20)  # 限制显示范围
        ax.hist(reviewer_counts[reviewer_counts <= max_reviewer], bins=range(0, max_reviewer + 2),
                color='#ff922b', edgecolor='black', alpha=0.8, align='left')
        ax.set_title('Reviewer Count Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Reviewer Count')
        ax.set_ylabel('PR Count')
        ax.set_xticks(range(0, max_reviewer + 2))

        plt.tight_layout()
        file_path = os.path.join(self.figures_dir, '04_reviewer_count_distribution.png')
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"图表已保存: {file_path}\n")

    def analyze_pr_length_distribution(self) -> None:
        """
        分析 5: PR 长度分布
        生成直方图
        """
        print("=" * 60)
        print("分析: PR 长度分布")
        print("=" * 60)

        if 'pr_length' not in self.df.columns:
            print("  缺少 pr_length 列\n")
            return

        pr_lengths = self.df['pr_length']
        print(f"  mean: {pr_lengths.mean():.2f}")
        print(f"  median: {pr_lengths.median():.2f}")
        print(f"  min: {pr_lengths.min()}")
        print(f"  max: {pr_lengths.max()}")
        print()

        # 直方图（对数尺度）
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 原始尺度
        upper = pr_lengths.quantile(0.95)
        axes[0].hist(pr_lengths[pr_lengths <= upper], bins=50,
                     color='#20c997', edgecolor='black', alpha=0.8)
        axes[0].axvline(pr_lengths.mean(), color='red', linestyle='--',
                        linewidth=2, label=f"mean={pr_lengths.mean():.0f}")
        axes[0].axvline(pr_lengths.median(), color='green', linestyle='--',
                        linewidth=2, label=f"median={pr_lengths.median():.0f}")
        axes[0].set_title('PR Length Distribution (P5-P95)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('PR Length (chars)')
        axes[0].set_ylabel('PR Count')
        axes[0].legend()

        # 对数尺度
        axes[1].hist(pr_lengths[pr_lengths > 0], bins=50,
                     color='#20c997', edgecolor='black', alpha=0.8)
        axes[1].set_xscale('log')
        axes[1].axvline(pr_lengths[pr_lengths > 0].mean(), color='red', linestyle='--',
                        linewidth=2, label=f"mean={pr_lengths.mean():.0f}")
        axes[1].axvline(pr_lengths.median(), color='green', linestyle='--',
                        linewidth=2, label=f"median={pr_lengths.median():.0f}")
        axes[1].set_title('PR Length Distribution (Log Scale)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('PR Length (chars, log scale)')
        axes[1].set_ylabel('PR Count')
        axes[1].legend()

        plt.tight_layout()
        file_path = os.path.join(self.figures_dir, '05_pr_length_distribution.png')
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"图表已保存: {file_path}\n")

    def analyze_ai_vs_human(self) -> None:
        """
        分析 6: AI 与 Human PR 数量比较
        包括 AI Reviewer 和 AI 生成代码两个维度
        """
        print("=" * 60)
        print("分析: AI vs Human PR 数量比较")
        print("=" * 60)

        # AI Reviewer 分析
        if 'has_ai_reviewer' in self.df.columns:
            ai_reviewer_counts = self.df['has_ai_reviewer'].value_counts()
            print(f"  AI Reviewer 参与的 PR: {ai_reviewer_counts.get(1, 0)}")
            print(f"  仅 Human Reviewer 的 PR: {ai_reviewer_counts.get(0, 0)}")
            print(f"  AI Reviewer 占比: {ai_reviewer_counts.get(1, 0) / len(self.df) * 100:.2f}%")
            print()

        # AI 生成代码分析
        if 'has_ai_generated_code' in self.df.columns:
            ai_code_counts = self.df['has_ai_generated_code'].value_counts()
            print(f"  含 AI 生成代码的 PR: {ai_code_counts.get(1, 0)}")
            print(f"  不含 AI 生成代码的 PR: {ai_code_counts.get(0, 0)}")
            print(f"  AI 生成代码占比: {ai_code_counts.get(1, 0) / len(self.df) * 100:.2f}%")
            print()

        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # AI Reviewer 饼图
        if 'has_ai_reviewer' in self.df.columns:
            ai_reviewer_counts = self.df['has_ai_reviewer'].value_counts()
            labels_reviewer = ['Human Only', 'With AI Reviewer']
            values_reviewer = [ai_reviewer_counts.get(0, 0), ai_reviewer_counts.get(1, 0)]
            colors_reviewer = ['#74c0fc', '#f06595']
            axes[0].pie(values_reviewer, labels=labels_reviewer, autopct='%1.1f%%',
                        colors=colors_reviewer, startangle=90, shadow=True, explode=(0, 0.05))
            axes[0].set_title('AI Reviewer Participation', fontsize=14, fontweight='bold')

        # AI 生成代码饼图
        if 'has_ai_generated_code' in self.df.columns:
            ai_code_counts = self.df['has_ai_generated_code'].value_counts()
            labels_code = ['Human Written', 'AI Generated']
            values_code = [ai_code_counts.get(0, 0), ai_code_counts.get(1, 0)]
            colors_code = ['#74c0fc', '#f06595']
            axes[1].pie(values_code, labels=labels_code, autopct='%1.1f%%',
                        colors=colors_code, startangle=90, shadow=True, explode=(0, 0.05))
            axes[1].set_title('AI Generated Code', fontsize=14, fontweight='bold')

        plt.tight_layout()
        file_path = os.path.join(self.figures_dir, '06_ai_vs_human.png')
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"图表已保存: {file_path}\n")

    def analyze_per_repo(self) -> None:
        """按仓库分析关键指标"""
        print("=" * 60)
        print("各仓库关键指标对比")
        print("=" * 60)

        repo_col = self.df['repo_name'] if 'repo_name' in self.df.columns else None
        if repo_col is None:
            return

        # 按仓库分组统计
        repo_stats = self.df.groupby('repo_name').agg({
            'pr_id': 'count',
            'is_merged': 'mean',
            'reviewer_count': 'mean',
            'total_comment_count': 'mean',
            'pr_length': 'mean',
            'files_changed': 'mean',
            'has_ai_reviewer': 'mean',
            'has_ai_generated_code': 'mean',
        }).round(3)

        repo_stats.columns = [
            'PR_Count', 'Merge_Rate', 'Avg_Reviewers',
            'Avg_Comments', 'Avg_PR_Length', 'Avg_Files_Changed',
            'AI_Reviewer_Ratio', 'AI_Code_Ratio'
        ]
        print(repo_stats.to_string())
        print()

        # 各仓库对比柱状图
        metrics = ['Merge_Rate', 'Avg_Reviewers', 'AI_Reviewer_Ratio', 'AI_Code_Ratio']
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            values = repo_stats[metric]
            colors = plt.cm.Set2(np.linspace(0, 1, len(values)))
            bars = ax.bar(range(len(values)), values.values, color=colors, edgecolor='black')
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(values.index, rotation=30, ha='right')
            ax.set_title(f'{metric} by Repository', fontsize=13, fontweight='bold')
            ax.set_ylabel(metric)

            for bar, v in zip(bars, values.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        file_path = os.path.join(self.figures_dir, '07_per_repo_comparison.png')
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"图表已保存: {file_path}\n")

    def analyze_correlations(self) -> None:
        """分析数值特征之间的相关性"""
        print("=" * 60)
        print("分析: 特征相关性")
        print("=" * 60)

        # 选择数值列
        numeric_cols = [
            'pr_length', 'title_length', 'files_changed',
            'additions', 'deletions', 'reviewer_count',
            'total_comment_count', 'label_count',
            'is_merged', 'has_ai_reviewer', 'has_ai_generated_code',
        ]
        numeric_cols = [c for c in numeric_cols if c in self.df.columns]

        if len(numeric_cols) < 2:
            print("  数值列不足\n")
            return

        corr_df = self.df[numeric_cols].corr()

        print("相关性矩阵:")
        print(corr_df.to_string())
        print()

        # 热力图
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)

        sns.heatmap(
            corr_df, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlBu_r', center=0, square=True,
            linewidths=1, cbar_kws={"shrink": 0.8},
            ax=ax
        )
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.xticks(rotation=30, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        file_path = os.path.join(self.figures_dir, '08_correlation_heatmap.png')
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"图表已保存: {file_path}\n")

    def analyze_language_distribution(self, top_n: int = 20) -> None:
        """
        分析: 编程语言分布
        统计所有 PR 中各编程语言出现的文件数量
        生成柱状图
        """
        print("=" * 60)
        print("分析: 编程语言分布")
        print("=" * 60)

        if 'languages' not in self.df.columns and 'primary_language' not in self.df.columns:
            print("  缺少语言相关列\n")
            return

        # 从所有 PR 中提取每个语言的出现次数（按文件数）
        all_languages = []
        for languages_str in self.df['languages']:
            if languages_str and isinstance(languages_str, str):
                for lang in languages_str.split(','):
                    lang = lang.strip()
                    if lang:
                        all_languages.append(lang)

        if not all_languages:
            print("  无语言数据\n")
            return

        lang_series = pd.Series(all_languages)
        lang_counts = lang_series.value_counts().head(top_n)

        print(f"  不同语言总数: {lang_series.nunique()}")
        print(f"  Top {top_n} 语言 (按出现PR数):")
        for lang, count in lang_counts.items():
            print(f"    {lang}: {count} PRs ({count / len(self.df) * 100:.1f}%)")

        # 按主语言统计 PR 数量
        if 'primary_language' in self.df.columns:
            primary_counts = self.df['primary_language'].value_counts().head(top_n)
            print(f"\n  Top {top_n} 主语言 (按PR主语言):")
            for lang, count in primary_counts.items():
                print(f"    {lang}: {count} PRs ({count / len(self.df) * 100:.1f}%)")
        print()

        # 绘制柱状图
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # 图表1: 按 PR 出现次数统计
        colors1 = plt.cm.tab20(np.linspace(0, 1, len(lang_counts)))
        bars1 = axes[0].barh(range(len(lang_counts)), lang_counts.values,
                             color=colors1, edgecolor='black', linewidth=0.8)
        axes[0].set_yticks(range(len(lang_counts)))
        axes[0].set_yticklabels(lang_counts.index)
        axes[0].set_xlabel('PR Count')
        axes[0].set_title(f'Programming Language Distribution by PR (Top {top_n})',
                          fontsize=13, fontweight='bold')
        axes[0].invert_yaxis()
        for bar, v in zip(bars1, lang_counts.values):
            axes[0].text(v + 0.5, bar.get_y() + bar.get_height() / 2,
                         str(v), va='center', fontsize=9)

        # 图表2: 按主语言统计
        if 'primary_language' in self.df.columns:
            colors2 = plt.cm.tab20b(np.linspace(0, 1, len(primary_counts)))
            bars2 = axes[1].barh(range(len(primary_counts)), primary_counts.values,
                                 color=colors2, edgecolor='black', linewidth=0.8)
            axes[1].set_yticks(range(len(primary_counts)))
            axes[1].set_yticklabels(primary_counts.index)
            axes[1].set_xlabel('PR Count')
            axes[1].set_title(f'Primary Language per PR (Top {top_n})',
                              fontsize=13, fontweight='bold')
            axes[1].invert_yaxis()
            for bar, v in zip(bars2, primary_counts.values):
                axes[1].text(v + 0.5, bar.get_y() + bar.get_height() / 2,
                             str(v), va='center', fontsize=9)

        plt.tight_layout()
        file_path = os.path.join(self.figures_dir, '09_language_distribution.png')
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"图表已保存: {file_path}\n")

    def run_all_analyses(self) -> None:
        """运行所有分析"""
        self.print_data_overview()
        self.analyze_merge_vs_non_merge()
        self.analyze_comment_distribution()
        self.analyze_label_distribution()
        self.analyze_reviewer_count_distribution()
        self.analyze_pr_length_distribution()
        self.analyze_language_distribution()
        self.analyze_ai_vs_human()
        self.analyze_per_repo()
        self.analyze_correlations()

        print("=" * 60)
        print("所有分析完成！图表已保存到:", self.figures_dir)
        print("=" * 60)


def load_dataset(csv_file: str = None) -> pd.DataFrame:
    """加载处理好的数据集"""
    if csv_file is None:
        csv_file = os.path.join(config.PROCESSED_DIR, "dataset.csv")

    if not os.path.exists(csv_file):
        print(f"数据集文件不存在: {csv_file}")
        print("请先运行 feature_extractor.py 构建数据集")
        return pd.DataFrame()

    df = pd.read_csv(csv_file)
    print(f"已加载数据集: {csv_file}")
    print(f"数据量: {df.shape[0]} 行, {df.shape[1]} 列")
    return df


if __name__ == "__main__":
    # 测试
    df = load_dataset()
    if not df.empty:
        analyzer = DataAnalyzer(df)
        analyzer.run_all_analyses()