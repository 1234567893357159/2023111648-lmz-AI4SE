"""
实验四 阶段二：代码上下文构建模块
针对每条 PR 数据，构造 4 种不同信息量的文本输入。
"""

import json
import os

import config

CONTEXT_TYPES = ["diff_only", "diff_pr_desc", "diff_commit", "diff_extra"]


def build_context(data, context_type):
    """
    根据 context_type 构建拼接好的文本上下文。

    参数:
        data: 单条 PR 数据（dict）
        context_type: "diff_only" | "diff_pr_desc" | "diff_commit" | "diff_extra"

    返回:
        str: 拼接好的上下文文本
    """
    diff = data.get("diff", "") or "(No diff available)"

    if context_type == "diff_only":
        return diff

    elif context_type == "diff_pr_desc":
        title = data.get("pr_title", "") or "(No title)"
        desc = data.get("pr_description", "") or "(No description provided)"
        parts = [
            "# Pull Request",
            f"## Title\n{title}",
            f"## Description\n{desc}",
            "## Code Changes (Diff)",
            diff,
        ]
        return "\n\n".join(parts)

    elif context_type == "diff_commit":
        commit_msg = data.get("commit_message", "") or "(No commit messages)"
        parts = [
            "# Commit Messages",
            commit_msg,
            "## Code Changes (Diff)",
            diff,
        ]
        return "\n\n".join(parts)

    elif context_type == "diff_extra":
        file_names = data.get("file_names", [])
        modified_functions = data.get("modified_functions", [])
        historical = data.get("historical_comments_text", "")

        parts = ["# Additional Context"]

        if file_names:
            parts.append(
                "## Changed Files\n" + "\n".join(f"  - {fn}" for fn in file_names)
            )

        if modified_functions:
            unique_funcs = list(dict.fromkeys(modified_functions))
            parts.append(
                "## Modified Functions\n"
                + "\n".join(f"  - {fn}" for fn in unique_funcs)
            )

        if historical.strip():
            parts.append("## Historical Comments\n" + historical)

        parts.append("## Code Changes (Diff)")
        parts.append(diff)
        return "\n\n".join(parts)

    else:
        raise ValueError(f"Unknown context_type: {context_type}")


def build_all_contexts(data):
    """为单条 PR 构建全部 4 种上下文"""
    result = {
        "pr_id": data["pr_id"],
        "repo": data["repo"],
        "contexts": {},
    }
    for ct in CONTEXT_TYPES:
        result["contexts"][ct] = build_context(data, ct)
    return result


def build_all_pr_contexts(data_list):
    """为所有 PR 批量构建全部上下文"""
    results = []
    for i, d in enumerate(data_list):
        results.append(build_all_contexts(d))
        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(data_list)} 条...")
    print(f"上下文构建完成，共 {len(results)} 条")
    return results


def save_contexts(contexts, path=None):
    """保存上下文数据到本地 JSON"""
    if path is None:
        path = config.CONTEXTS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(contexts, f, indent=2, ensure_ascii=False)
    print(f"上下文数据已保存到: {path}")
    print(f"文件大小: {os.path.getsize(path) / 1024 / 1024:.2f} MB")


def compute_context_stats(contexts):
    """统计各上下文类型的长度分布"""
    stats = {}
    for ct in CONTEXT_TYPES:
        lengths = [len(c["contexts"][ct]) for c in contexts]
        stats[ct] = {
            "count": len(lengths),
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "total_chars": sum(lengths),
        }
    return stats


def print_context_stats(stats):
    """打印上下文统计信息"""
    print("=" * 60)
    print("各上下文类型长度统计")
    print("=" * 60)
    for ct in CONTEXT_TYPES:
        s = stats[ct]
        print(f"\n{ct}:")
        print(f"  样本数: {s['count']}")
        print(f"  最小长度: {s['min']:,} 字符")
        print(f"  最大长度: {s['max']:,} 字符")
        print(f"  平均长度: {s['mean']:,.0f} 字符")
        print(f"  总字符数: {s['total_chars']:,} 字符")


def print_context_sample(contexts, idx):
    """打印一条 PR 的 4 种上下文样例（截断显示）"""
    c = contexts[idx]
    print(f"--- PR #{c['pr_id']} ({c['repo']}) ---")
    for ct in CONTEXT_TYPES:
        text = c["contexts"][ct]
        preview = text[:300].replace("\n", "\\n")
        print(f"\n  [{ct}] 长度: {len(text):,} 字符")
        print(f"  预览: {preview}...")
    print()


def run_context_building(data_list):
    """执行完整的上下文构建流程，返回 (contexts, stats)"""
    print("=" * 60)
    print("阶段二：构建代码上下文")
    print("=" * 60)
    print(f"4 种上下文类型: {CONTEXT_TYPES}")
    print()

    contexts = build_all_pr_contexts(data_list)
    save_contexts(contexts)

    stats = compute_context_stats(contexts)

    return contexts, stats