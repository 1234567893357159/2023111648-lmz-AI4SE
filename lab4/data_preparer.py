"""
实验四 数据准备模块
负责从 lab1 原始数据中加载、筛选、随机抽取 PR，并提取 lab4 所需字段。
"""

import json
import os
import random

import config


def load_all_prs():
    """加载 lab1 中所有仓库的 PR 原始数据"""
    all_prs = []
    for owner, repo_name in config.TARGET_REPOS:
        repo_key = f"{owner}_{repo_name}"
        json_path = os.path.join(config.LAB1_RAW_DIR, f"{repo_key}_pulls.json")
        if not os.path.exists(json_path):
            print(f"  ⚠️ 文件不存在: {json_path}")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            prs = json.load(f)
        for pr in prs:
            pr["repo_key"] = repo_key
        all_prs.extend(prs)
        print(f"  加载 {repo_key}: {len(prs)} PRs")
    print(f"\n  共加载 {len(all_prs)} 个 PR")
    return all_prs


def has_review_comments(pr):
    """判断 PR 是否有实质性的 review comment（非空、非 bot）"""
    reviews = pr.get("reviews", [])
    has_review = any(
        r.get("body", "").strip()
        for r in reviews
    )

    review_comments = pr.get("review_comments", [])
    has_inline = any(
        rc.get("body", "").strip()
        for rc in review_comments
    )

    issue_comments = pr.get("issue_comments", [])
    human_comments = [
        c for c in issue_comments
        if c.get("body", "").strip()
        and "bot" not in (c.get("user") or "").lower()
    ]
    has_issue = len(human_comments) > 0

    return has_review or has_inline or has_issue


def filter_prs_with_comments(all_prs):
    """筛选有 Review Comment 的 PR"""
    filtered = [pr for pr in all_prs if has_review_comments(pr)]
    print(f"有 Review Comment 的 PR: {len(filtered)} / {len(all_prs)}")
    print(f"占比: {len(filtered) / len(all_prs) * 100:.1f}%")
    return filtered


def random_select_prs(filtered_prs, count=None):
    """随机抽取 PR"""
    if count is None:
        count = config.SELECT_COUNT
    count = min(count, len(filtered_prs))
    selected = random.sample(filtered_prs, count)
    print(f"随机抽取 {len(selected)} 个 PR")
    return selected


def extract_fields(pr):
    """从原始 PR 数据中提取 lab4 需要的字段"""
    files = pr.get("files", [])
    commits = pr.get("commits", [])

    diff_parts = []
    for f in files:
        filename = f.get("filename", "")
        patch = f.get("patch", "")
        if patch:
            diff_parts.append(f"--- a/{filename}\n+++ b/{filename}\n{patch}")
    diff = "\n".join(diff_parts)

    commit_msgs = [c.get("message", "") for c in commits]
    commit_message = "\n---\n".join(commit_msgs)

    reviews = pr.get("reviews", [])
    review_comments = pr.get("review_comments", [])
    issue_comments = pr.get("issue_comments", [])

    file_names = [f.get("filename", "") for f in files]

    modified_functions = []
    for f in files:
        funcs = f.get("modified_functions", [])
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


def extract_all_fields(selected_prs):
    """批量提取所有选中 PR 的字段"""
    data = [extract_fields(pr) for pr in selected_prs]
    print(f"提取完成，共 {len(data)} 条数据")
    return data


def save_selected_data(data, path=None):
    """保存挑选后的数据到 JSON"""
    if path is None:
        path = config.SELECTED_PRS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到: {path}")
    print(f"文件大小: {os.path.getsize(path) / 1024 / 1024:.2f} MB")


def compute_summary(data):
    """计算数据集的统计摘要"""
    repo_counts = {}
    merge_counts = {"merged": 0, "not_merged": 0}
    total_comments = 0
    total_reviewers = set()

    for d in data:
        repo = d["repo"]
        repo_counts[repo] = repo_counts.get(repo, 0) + 1
        if d["label"] == 1:
            merge_counts["merged"] += 1
        else:
            merge_counts["not_merged"] += 1
        total_comments += len(d["review_comments"]) + len(d["issue_comments"])
        for r in d["reviews"]:
            if r["reviewer"]:
                total_reviewers.add(r["reviewer"])

    diff_lengths = [len(d["diff"]) for d in data]
    body_lengths = [len(d["pr_description"]) for d in data]

    return {
        "total_prs": len(data),
        "repo_distribution": repo_counts,
        "merge_distribution": merge_counts,
        "merge_rate": merge_counts["merged"] / len(data) if data else 0,
        "total_comments": total_comments,
        "avg_comments_per_pr": total_comments / len(data) if data else 0,
        "unique_reviewers": len(total_reviewers),
        "diff_length_stats": {
            "min": min(diff_lengths),
            "max": max(diff_lengths),
            "mean": sum(diff_lengths) / len(diff_lengths),
        },
        "body_length_stats": {
            "min": min(body_lengths),
            "max": max(body_lengths),
            "mean": sum(body_lengths) / len(body_lengths),
        },
    }


def print_sample(data, idx):
    """打印一条数据样例"""
    d = data[idx]
    print(f"--- 样例 {idx + 1} ---")
    print(f"  PR ID: {d['pr_id']}")
    print(f"  仓库: {d['repo']}")
    print(f"  标题: {d['pr_title'][:100]}")
    print(f"  是否合并: {'✅ 是' if d['label'] == 1 else '❌ 否'}")
    print(f"  修改文件数: {d['files_changed']}")
    print(f"  +{d['additions']} / -{d['deletions']}")
    print(f"  Reviews 数: {len(d['reviews'])}")
    print(f"  行内评论数: {len(d['review_comments'])}")
    print(f"  Issue 评论数: {len(d['issue_comments'])}")
    print(f"  PR 描述长度: {len(d['pr_description'])} 字符")
    print(f"  Diff 长度: {len(d['diff'])} 字符")
    print(f"  Commit 数量: {d['commit_message'].count('---') + 1}")
    print()


def run_data_preparation():
    """执行完整的数据准备流程，返回 (data, summary)"""
    random.seed(config.RANDOM_SEED)
    print(f"随机种子: {config.RANDOM_SEED}")
    print(f"挑选数量: {config.SELECT_COUNT}")
    print(f"Lab1 数据目录: {config.LAB1_RAW_DIR}")
    print(f"Lab4 输出目录: {config.LAB4_DATA_DIR}")
    print()

    print("=" * 60)
    print("步骤一：加载 Lab1 全部原始数据")
    print("=" * 60)
    all_prs = load_all_prs()

    print()
    print("=" * 60)
    print("步骤二：筛选有 Review Comment 的 PR")
    print("=" * 60)
    filtered = filter_prs_with_comments(all_prs)

    print()
    print("=" * 60)
    print("步骤三：随机抽取")
    print("=" * 60)
    selected = random_select_prs(filtered)

    print()
    print("=" * 60)
    print("步骤四：提取 Lab4 所需字段")
    print("=" * 60)
    data = extract_all_fields(selected)

    print()
    print("=" * 60)
    print("步骤五：保存数据")
    print("=" * 60)
    save_selected_data(data)

    print()
    print("=" * 60)
    print("步骤六：计算统计摘要")
    print("=" * 60)
    summary = compute_summary(data)
    save_summary(summary)

    return data, summary


def save_summary(summary, path=None):
    """保存统计摘要"""
    if path is None:
        path = config.SUMMARY_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"统计摘要已保存到: {path}")