"""
实验六 步骤二：上下文构建模块
构建 5 种不同粒度的软件工程上下文：
  - diff_only:      仅代码 Diff
  - diff_pr_desc:   Diff + PR 描述
  - diff_repo:      Diff + Repository 上下文（文件信息 + 调用关系）
  - diff_issue:     Diff + Issue 信息
  - diff_full:      完整软件工程上下文（融合所有信息）
"""

import json
import os

import config


CONTEXT_TYPES = ["diff_only", "diff_pr_desc", "diff_repo", "diff_issue", "diff_full"]


def build_diff_only(data):
    """仅返回代码 diff"""
    diff = data.get("diff", "") or "(No diff available)"
    return diff


def build_diff_pr_desc(data):
    """Diff + PR 标题 + PR 描述"""
    diff = data.get("diff", "") or "(No diff available)"
    title = data.get("pr_title", "") or "(No title)"
    desc = data.get("pr_description", "") or "(No description provided)"

    return "\n\n".join([
        "# Pull Request",
        f"## Title\n{title}",
        f"## Description\n{desc}",
        "## Code Changes (Diff)",
        diff,
    ])


def build_diff_repo(data):
    """Diff + Repository 上下文（文件信息 + 调用关系）"""
    diff = data.get("diff", "") or "(No diff available)"
    files_with_content = data.get("files_with_content", [])
    related_functions = data.get("related_functions", {})

    parts = ["# Repository Context"]

    # ---- 修改文件列表 ----
    if files_with_content:
        lines = []
        for fwc in files_with_content:
            filename = fwc.get("filename", "")
            language = fwc.get("language", "Unknown")
            status = fwc.get("status", "")
            adds = fwc.get("additions", 0)
            dels = fwc.get("deletions", 0)
            lines.append(f"  - {filename}  [{language}]  {status}  (+{adds} -{dels})")
        parts.append("## Changed Files\n" + "\n".join(lines))

    # ---- 修改函数 ----
    modified_functions = data.get("modified_functions", [])
    if modified_functions:
        unique_funcs = list(dict.fromkeys(modified_functions))
        parts.append(
            "## Modified Functions\n" +
            "\n".join(f"  - {fn}" for fn in unique_funcs)
        )

    # ---- 外部依赖 ----
    imports = related_functions.get("imports", [])
    if imports:
        parts.append(
            "## External Dependencies (Imports)\n" +
            "\n".join(f"  - {imp}" for imp in imports)
        )

    # ---- 函数调用关系 ----
    callees = related_functions.get("callees", [])
    if callees:
        parts.append(
            "## Function Call Relationships\n"
            "  Called functions (not modified in this PR):\n" +
            "\n".join(f"  - {c}" for c in callees)
        )

    # ---- diff ----
    parts.append("## Code Changes (Diff)")
    parts.append(diff)

    return "\n\n".join(parts)


def build_diff_issue(data):
    """Diff + Issue 信息"""
    diff = data.get("diff", "") or "(No diff available)"
    issue_references = data.get("issue_references", [])
    issue_comments = data.get("issue_comments", [])

    parts = ["# Issue Context"]

    # ---- Issue 引用 ----
    if issue_references:
        ref_lines = []
        for ref in issue_references:
            ref_lines.append(f"### Issue #{ref.get('issue_number', '?')}")
            ref_lines.append(f"Source: {ref.get('source', 'unknown')}")
            ref_lines.append(f"Context: {ref.get('context', '')}")
            ref_lines.append("")
        parts.append("## Referenced Issues\n" + "\n".join(ref_lines))
    else:
        parts.append("## Referenced Issues\n(No issue references found in this PR)")

    # ---- 相关讨论 ----
    if issue_comments:
        discussion_parts = []
        for ic in issue_comments:
            user = ic.get("user", "unknown")
            body = ic.get("body", "")
            if body.strip():
                discussion_parts.append(f"**{user}**: {body.strip()}")
        if discussion_parts:
            combined = "\n\n".join(discussion_parts)
            if len(combined) > 2000:
                combined = combined[:2000] + "\n...(truncated)"
            parts.append("## Related Discussions\n" + combined)

    # ---- diff ----
    parts.append("## Code Changes (Diff)")
    parts.append(diff)

    return "\n\n".join(parts)


def build_diff_full(data):
    """完整软件工程上下文（融合所有信息）"""
    diff = data.get("diff", "") or "(No diff available)"
    title = data.get("pr_title", "") or "(No title)"
    desc = data.get("pr_description", "") or "(No description provided)"
    files_with_content = data.get("files_with_content", [])
    related_functions = data.get("related_functions", {})
    issue_references = data.get("issue_references", [])
    issue_comments = data.get("issue_comments", [])
    historical = data.get("historical_comments_text", "")

    parts = ["# Full Software Engineering Context"]

    # ---- PR 信息 ----
    parts.append("## Pull Request")
    parts.append(f"### Title\n{title}")
    parts.append(f"### Description\n{desc}")

    # ---- Repository 上下文 ----
    parts.append("## Repository Context")

    if files_with_content:
        lines = []
        for fwc in files_with_content:
            filename = fwc.get("filename", "")
            language = fwc.get("language", "Unknown")
            status = fwc.get("status", "")
            adds = fwc.get("additions", 0)
            dels = fwc.get("deletions", 0)
            lines.append(f"  - {filename}  [{language}]  {status}  (+{adds} -{dels})")
        parts.append("### Changed Files\n" + "\n".join(lines))

    modified_functions = data.get("modified_functions", [])
    if modified_functions:
        unique_funcs = list(dict.fromkeys(modified_functions))
        parts.append(
            "### Modified Functions\n" +
            "\n".join(f"  - {fn}" for fn in unique_funcs)
        )

    imports = related_functions.get("imports", [])
    if imports:
        parts.append(
            "### External Dependencies\n" +
            "\n".join(f"  - {imp}" for imp in imports)
        )

    callees = related_functions.get("callees", [])
    if callees:
        parts.append(
            "### Function Call Relationships\n" +
            "\n".join(f"  - {c}" for c in callees)
        )

    # ---- Issue 上下文 ----
    parts.append("## Issue Context")
    if issue_references:
        ref_lines = []
        for ref in issue_references:
            ref_lines.append(f"### Issue #{ref.get('issue_number', '?')}")
            ref_lines.append(f"Source: {ref.get('source', 'unknown')}")
            ref_lines.append(f"Context: {ref.get('context', '')}")
            ref_lines.append("")
        parts.append("### Referenced Issues\n" + "\n".join(ref_lines))
    else:
        parts.append("### Referenced Issues\n(No issue references found)")

    if issue_comments:
        discussion_parts = []
        for ic in issue_comments:
            user = ic.get("user", "unknown")
            body = ic.get("body", "")
            if body.strip():
                discussion_parts.append(f"**{user}**: {body.strip()}")
        if discussion_parts:
            combined = "\n\n".join(discussion_parts)
            if len(combined) > 1500:
                combined = combined[:1500] + "\n...(truncated)"
            parts.append("### Related Discussions\n" + combined)

    # ---- 历史评论 ----
    if historical.strip():
        parts.append("## Historical Review Comments")
        if len(historical) > 1500:
            historical = historical[:1500] + "\n...(truncated)"
        parts.append(historical)

    # ---- diff ----
    parts.append("## Code Changes (Diff)")
    parts.append(diff)

    return "\n\n".join(parts)


def build_context(data, context_type):
    """根据 context_type 构建拼接好的文本上下文"""
    builders = {
        "diff_only": build_diff_only,
        "diff_pr_desc": build_diff_pr_desc,
        "diff_repo": build_diff_repo,
        "diff_issue": build_diff_issue,
        "diff_full": build_diff_full,
    }
    builder = builders.get(context_type)
    if builder is None:
        raise ValueError(f"Unknown context_type: {context_type}")
    return builder(data)


def build_all_contexts(data):
    """为单条 PR 构建全部 5 种上下文"""
    result = {
        "pr_id": data["pr_id"],
        "repo": data["repo"],
        "label": data["label"],
        "contexts": {},
    }
    for ct in CONTEXT_TYPES:
        result["contexts"][ct] = build_context(data, ct)
    return result


def build_all_pr_contexts(data_list):
    """为所有 PR 批量构建全部上下文"""
    results = []
    total = len(data_list)
    for i, d in enumerate(data_list):
        results.append(build_all_contexts(d))
        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{total} 条...")
    print(f"  上下文构建完成，共 {len(results)} 条")
    return results


def save_contexts(contexts):
    """保存上下文数据到 JSON"""
    os.makedirs(os.path.dirname(config.CONTEXTS_PATH), exist_ok=True)
    with open(config.CONTEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(contexts, f, indent=2, ensure_ascii=False)
    file_size = os.path.getsize(config.CONTEXTS_PATH) / 1024 / 1024
    print(f"  上下文数据已保存到: {config.CONTEXTS_PATH}")
    print(f"  文件大小: {file_size:.2f} MB")


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
    header = f"{'上下文':<20} {'样本数':<8} {'最小':<8} {'最大':<8} {'平均':<10} {'总字符':<12}"
    print(header)
    print("-" * 60)
    for ct in CONTEXT_TYPES:
        s = stats[ct]
        print(f"{ct:<20} {s['count']:<8} {s['min']:<8,} {s['max']:<8,} "
              f"{s['mean']:<10,.0f} {s['total_chars']:<12,}")


def print_context_sample(contexts, idx=None):
    """打印一条 PR 的五种上下文样例（截断显示）"""
    if idx is None:
        idx = min(2, len(contexts) - 1)
    c = contexts[idx]
    print(f"\n{'=' * 60}")
    print(f"上下文样例 — PR #{c['pr_id']} ({c['repo']})  label={c['label']}")
    print(f"{'=' * 60}")
    for ct in CONTEXT_TYPES:
        text = c["contexts"][ct]
        preview = text[:200].replace("\n", "\\n")
        print(f"\n[{ct}]  长度: {len(text):,} 字符")
        print(f"  预览: {preview}...")


def run_context_building(data_list):
    """执行完整的上下文构建流程，返回 (contexts, stats)"""
    print("=" * 60)
    print("  实验六 步骤二：上下文构建")
    print("=" * 60)
    print(f"  5 种上下文类型: {CONTEXT_TYPES}")
    print(f"  输入 PR 数: {len(data_list)}")
    print()

    contexts = build_all_pr_contexts(data_list)
    save_contexts(contexts)

    stats = compute_context_stats(contexts)

    return contexts, stats


if __name__ == "__main__":
    with open(config.SELECTED_PRS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    contexts, stats = run_context_building(data)
    print_context_stats(stats)
    print_context_sample(contexts)