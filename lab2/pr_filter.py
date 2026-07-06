"""
PR 筛选模块
从 lab1 原始数据中筛选人类编写的 PR，保存到本地
"""

import json
import os
import csv
from typing import Dict, List, Tuple

import config


class PRFilter:
    """PR 筛选器"""

    def __init__(self):
        self.stats = {}

    def load_raw_prs(self, owner: str, repo_name: str) -> List[Dict]:
        """从 lab1 加载原始 PR 数据"""
        file_path = os.path.join(
            config.LAB1_RAW_DIR,
            f"{owner}_{repo_name}_pulls.json"
        )
        if not os.path.exists(file_path):
            print(f"  警告: 文件不存在 {file_path}")
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def is_bot_author(self, author: str) -> bool:
        """判断是否为 Bot 作者"""
        author_lower = author.lower() if author else ""
        return any(kw in author_lower for kw in config.BOT_AUTHOR_KEYWORDS)

    def detect_ai_generated_code(self, pr_data: Dict) -> Tuple[int, str]:
        """
        检测是否包含 AI 生成代码
        复用 lab1 的三级检测策略：
        1. Bot 作者 → 直接排除
        2. 关键词匹配
        3. 启发式规则

        返回: (has_ai_code, indicators)
        """
        commits = pr_data.get("commits", [])
        body = (pr_data.get("body") or "").lower()
        title = (pr_data.get("title") or "").lower()
        author = (pr_data.get("author") or "").lower()
        labels = [lbl.lower() for lbl in pr_data.get("labels", [])]

        has_ai_code = 0
        ai_indicators = []

        if self.is_bot_author(pr_data.get("author", "")):
            return 0, "bot_excluded"

        all_commit_text = " ".join(
            c.get("message", "") for c in commits
        ).lower()
        search_texts = {
            "commit": all_commit_text,
            "body": body,
            "title": title,
            "labels": " ".join(labels),
        }

        for source, text in search_texts.items():
            if not text:
                continue
            for keyword in config.AI_KEYWORDS:
                if keyword in text:
                    has_ai_code = 1
                    ai_indicators.append(f"{source}: {keyword}")
                    break

        if has_ai_code == 0:
            files_changed = pr_data.get("changed_files", 0)
            additions = pr_data.get("additions", 0)
            first_commit = commits[0].get("message", "") if commits else ""

            if files_changed <= 2 and additions >= 200:
                if len(first_commit.strip()) <= 30:
                    has_ai_code = 1
                    ai_indicators.append(
                        "heuristic: large_single_file_short_commit"
                    )

            if has_ai_code == 0:
                template_markers = [
                    "## summary", "## changes", "## testing",
                    "## description", "## motivation",
                ]
                match_count = sum(1 for m in template_markers if m in body)
                if match_count >= 3 and len(body) > 500:
                    has_ai_code = 1
                    ai_indicators.append("heuristic: template_body")

        return has_ai_code, "; ".join(ai_indicators) if ai_indicators else ""

    def filter_prs(self, owner: str, repo_name: str) -> Dict:
        """
        筛选单个仓库的 PR
        返回: {"human": [...], "non_human": [...], "bot": [...], "stats": {...}}
        """
        prs = self.load_raw_prs(owner, repo_name)
        human_prs = []
        non_human_prs = []
        bot_prs = []

        for pr in prs:
            if self.is_bot_author(pr.get("author", "")):
                bot_prs.append(pr)
                continue

            has_ai, indicators = self.detect_ai_generated_code(pr)
            pr["has_ai_generated_code"] = has_ai
            pr["ai_code_indicators"] = indicators

            if has_ai == 1:
                non_human_prs.append(pr)
            else:
                human_prs.append(pr)

        stats = {
            "repo": f"{owner}/{repo_name}",
            "total": len(prs),
            "human": len(human_prs),
            "non_human": len(non_human_prs),
            "bot": len(bot_prs),
        }

        return {
            "human": human_prs,
            "non_human": non_human_prs,
            "bot": bot_prs,
            "stats": stats,
        }

    def save_results(self, owner: str, repo_name: str, results: Dict):
        """保存筛选结果到本地"""
        repo_key = f"{owner}_{repo_name}"

        human_file = os.path.join(config.HUMAN_DIR, f"{repo_key}_pulls.json")
        with open(human_file, "w", encoding="utf-8") as f:
            json.dump(results["human"], f, ensure_ascii=False, indent=2)

        non_human_file = os.path.join(config.NON_HUMAN_DIR, f"{repo_key}_pulls.json")
        with open(non_human_file, "w", encoding="utf-8") as f:
            json.dump(results["non_human"], f, ensure_ascii=False, indent=2)

    def run_all(self) -> Dict:
        """运行所有仓库的筛选"""
        all_stats = []
        total_human = 0
        total_non_human = 0
        total_bot = 0
        total_all = 0

        print("=" * 60)
        print("开始筛选人类编写的 PR...")
        print("=" * 60)

        for owner, repo_name in config.TARGET_REPOS:
            print(f"\n处理 {owner}/{repo_name}...")
            results = self.filter_prs(owner, repo_name)
            stats = results["stats"]
            all_stats.append(stats)

            total_human += stats["human"]
            total_non_human += stats["non_human"]
            total_bot += stats["bot"]
            total_all += stats["total"]

            print(f"  总计: {stats['total']}, 人类: {stats['human']}, "
                  f"AI: {stats['non_human']}, Bot: {stats['bot']}")

            self.save_results(owner, repo_name, results)

        summary = {
            "stats": all_stats,
            "total_all": total_all,
            "total_human": total_human,
            "total_non_human": total_non_human,
            "total_bot": total_bot,
        }

        print(f"\n{'=' * 60}")
        print(f"筛选完成!")
        print(f"  总 PR 数: {total_all}")
        print(f"  人类编写: {total_human}")
        print(f"  AI 生成:  {total_non_human}")
        print(f"  Bot 提交: {total_bot}")
        print(f"{'=' * 60}")

        return summary

    def save_filtered_csv(self):
        """将所有人类 PR 信息保存为 CSV 汇总"""
        all_human = []
        for owner, repo_name in config.TARGET_REPOS:
            repo_key = f"{owner}_{repo_name}"
            human_file = os.path.join(config.HUMAN_DIR, f"{repo_key}_pulls.json")
            if os.path.exists(human_file):
                with open(human_file, "r", encoding="utf-8") as f:
                    human_prs = json.load(f)
                    for pr in human_prs:
                        all_human.append({
                            "pr_id": pr.get("pr_id"),
                            "repo_owner": owner,
                            "repo_name": repo_name,
                            "title": pr.get("title"),
                            "author": pr.get("author"),
                            "created_at": pr.get("created_at"),
                            "additions": pr.get("additions", 0),
                            "deletions": pr.get("deletions", 0),
                            "changed_files": pr.get("changed_files", 0),
                            "is_merged": 1 if pr.get("merged") else 0,
                            "label_count": len(pr.get("labels", [])),
                            "labels": ",".join(pr.get("labels", [])),
                        })

        csv_path = os.path.join(config.DATA_DIR, "filtered_human_prs.csv")
        if all_human:
            fieldnames = list(all_human[0].keys())
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_human)
            print(f"人类 PR 汇总已保存到: {csv_path} (共 {len(all_human)} 条)")


if __name__ == "__main__":
    filter_obj = PRFilter()
    summary = filter_obj.run_all()
    filter_obj.save_filtered_csv()