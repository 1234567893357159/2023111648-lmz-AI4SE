"""
步骤四：AI PR 数据准备模块
加载 step1 筛选出的 AI 生成 PR，提取字段，构建 4 种上下文
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "lab4"))

import config


CONTEXT_TYPES = ["diff_only", "diff_pr_desc", "diff_commit", "diff_extra"]


def load_ai_prs():
    """加载 step1 筛选出的所有 AI PR"""
    all_prs = []
    for owner, repo_name in config.TARGET_REPOS:
        repo_key = f"{owner}_{repo_name}"
        json_path = os.path.join(config.AI_PULLS_DIR, f"{repo_key}_pulls.json")
        if not os.path.exists(json_path):
            print(f"  [跳过] 文件不存在: {json_path}")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            prs = json.load(f)
        for pr in prs:
            pr["repo_key"] = repo_key
        all_prs.extend(prs)
        print(f"  加载 {repo_key}: {len(prs)} 个 AI PR")
    print(f"  共加载 {len(all_prs)} 个 AI PR")
    return all_prs


def extract_fields(pr):
    """从原始 PR 数据中提取 lab4 需要的字段"""
    files = pr.get("files", [])
    commits = pr.get("commits", [])

    diff_parts = []
    for f_item in files:
        filename = f_item.get("filename", "")
        patch = f_item.get("patch", "")
        if patch:
            diff_parts.append(f"--- a/{filename}\n+++ b/{filename}\n{patch}")
    diff = "\n".join(diff_parts)

    commit_msgs = [c.get("message", "") for c in commits]
    commit_message = "\n---\n".join(commit_msgs)

    reviews = pr.get("reviews", [])
    review_comments = pr.get("review_comments", [])
    issue_comments = pr.get("issue_comments", [])

    file_names = [f_item.get("filename", "") for f_item in files]

    modified_functions = []
    for f_item in files:
        funcs = f_item.get("modified_functions", [])
        if isinstance(funcs, list):
            for fn in funcs:
                if isinstance(fn, str) and fn.strip():
                    modified_functions.append(fn.strip())

    historical_parts = []
    for rc in review_comments:
        body = rc.get("body", "").strip()
        if body:
            historical_parts.append(
                f"[Inline Review by {rc.get('user', 'unknown')}] {body}"
            )
    for ic in issue_comments:
        body = ic.get("body", "").strip()
        if body:
            historical_parts.append(
                f"[Issue Comment by {ic.get('user', 'unknown')}] {body}"
            )
    historical_comments_text = "\n".join(historical_parts)

    return {
        "pr_id": pr["pr_id"],
        "repo": pr["repo_key"],
        "pr_title": pr.get("title", ""),
        "pr_description": pr.get("body", "") or "",
        "diff": diff,
        "commit_message": commit_message,
        "label": 1 if pr.get("merged") else 0,
        "files_changed": pr.get("changed_files", 0),
        "additions": pr.get("additions", 0),
        "deletions": pr.get("deletions", 0),
        "file_names": file_names,
        "modified_functions": modified_functions,
        "historical_comments_text": historical_comments_text,
        "reviews": [
            {
                "reviewer": r.get("reviewer", ""),
                "state": r.get("state", ""),
                "body": r.get("body", ""),
            }
            for r in reviews
        ],
        "review_comments": [
            {
                "user": rc.get("user", ""),
                "body": rc.get("body", ""),
                "path": rc.get("path", ""),
            }
            for rc in review_comments
        ],
        "issue_comments": [
            {
                "user": ic.get("user", ""),
                "body": ic.get("body", ""),
            }
            for ic in issue_comments
        ],
    }


def build_context(data, context_type):
    """根据 context_type 构建拼接好的文本上下文"""
    diff = data.get("diff", "") or "(No diff available)"

    if context_type == "diff_only":
        return diff

    elif context_type == "diff_pr_desc":
        title = data.get("pr_title", "") or "(No title)"
        desc = data.get("pr_description", "") or "(No description provided)"
        return "\n\n".join([
            "# Pull Request",
            f"## Title\n{title}",
            f"## Description\n{desc}",
            "## Code Changes (Diff)",
            diff,
        ])

    elif context_type == "diff_commit":
        commit_msg = data.get("commit_message", "") or "(No commit messages)"
        return "\n\n".join([
            "# Commit Messages",
            commit_msg,
            "## Code Changes (Diff)",
            diff,
        ])

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
    return {
        "pr_id": data["pr_id"],
        "repo": data["repo"],
        "label": data["label"],
        "contexts": {ct: build_context(data, ct) for ct in CONTEXT_TYPES},
    }


def build_all_pr_contexts(data_list):
    """为所有 PR 批量构建全部上下文"""
    results = []
    for i, d in enumerate(data_list):
        results.append(build_all_contexts(d))
        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(data_list)} 条...")
    print(f"  上下文构建完成，共 {len(results)} 条")
    return results


def compute_summary(data_list):
    """计算数据集的统计摘要"""
    repo_counts = {}
    merge_counts = {"merged": 0, "not_merged": 0}
    total_comments = 0

    for d in data_list:
        repo = d["repo"]
        repo_counts[repo] = repo_counts.get(repo, 0) + 1
        if d["label"] == 1:
            merge_counts["merged"] += 1
        else:
            merge_counts["not_merged"] += 1
        total_comments += len(d.get("review_comments", [])) + len(d.get("issue_comments", []))

    diff_lengths = [len(d["diff"]) for d in data_list]

    return {
        "total_prs": len(data_list),
        "merged": merge_counts["merged"],
        "not_merged": merge_counts["not_merged"],
        "repo_counts": repo_counts,
        "total_comments": total_comments,
        "avg_diff_length": sum(diff_lengths) / len(diff_lengths) if diff_lengths else 0,
    }


class AIPRDataPreparer:
    """AI PR 数据准备器"""

    def __init__(self, max_prs=100, seed=42):
        self.max_prs = max_prs
        self.seed = seed
        self.raw_prs = []
        self.extracted_prs = []
        self.contexts = []
        self.summary = {}

    def run_all(self):
        """执行完整数据准备流程"""
        print("=" * 60)
        print("步骤四数据准备: 加载 AI PR → 提取字段 → 构建上下文")
        print("=" * 60)

        print("\n[1/4] 加载 AI PR 数据...")
        self.raw_prs = load_ai_prs()

        print(f"\n[2/4] 随机采样 {self.max_prs} 条 PR...")
        random.seed(self.seed)
        if len(self.raw_prs) > self.max_prs:
            self.raw_prs = random.sample(self.raw_prs, self.max_prs)
        print(f"  采样完成，共 {len(self.raw_prs)} 条")

        print("\n[3/4] 提取字段...")
        self.extracted_prs = [extract_fields(pr) for pr in self.raw_prs]
        print(f"  提取完成，共 {len(self.extracted_prs)} 条")

        print("\n[4/4] 构建 4 种上下文...")
        self.contexts = build_all_pr_contexts(self.extracted_prs)

        self.summary = compute_summary(self.extracted_prs)
        print(f"\n摘要: {self.summary['total_prs']} 个 PR, "
              f"merged={self.summary['merged']}, "
              f"not_merged={self.summary['not_merged']}")

        return self.contexts, self.summary


if __name__ == "__main__":
    preparer = AIPRDataPreparer()
    contexts, summary = preparer.run_all()