"""
GitHub API 数据获取模块
通过 PyGithub 获取 Pull Request、Review、Comment、Commit 等数据
"""

import json
import re
import certifi
import time
import os
from typing import Dict, List, Optional, Tuple

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import requests
requests.utils.DEFAULT_CA_BUNDLE_PATH = certifi.where()

from github import Github, GithubException, PullRequest
from github.Organization import Organization
from github.Repository import Repository
from github.IssueComment import IssueComment
from github.Commit import Commit

import config

class GithubDataFetcher:
    """GitHub 数据获取器"""

    def __init__(self, token: str = None):
        """初始化"""
        if token is None:
            token = config.GITHUB_TOKEN
        self.gh = Github(token, per_page=100)
        self._rate_limit_info()

    def _rate_limit_info(self) -> None:
        """打印限流信息"""
        rate_limit = self.gh.get_rate_limit()
        remaining = rate_limit.core.remaining
        reset_time = rate_limit.core.reset
        print(f"GitHub API Rate Limit: {remaining} requests remaining")
        print(f"Reset at: {reset_time}")

    @staticmethod
    def _retry_api_call(call_fn, *args, max_retries: int = 3, retry_delay: int = 10, **kwargs):
        """
        通用 API 调用重试机制
        当遇到 GithubException（如 403、404）或其他异常时，自动暂停重试
        :param call_fn: 要执行的 API 调用函数
        :param max_retries: 最大重试次数（仅关键字参数）
        :param retry_delay: 每次重试前的暂停秒数（仅关键字参数）
        :return: API 调用结果，若多次重试失败则返回 None
        """
        for attempt in range(1, max_retries + 1):
            try:
                return call_fn(*args, **kwargs)
            except GithubException as e:
                status = e.status if hasattr(e, 'status') else '?'
                if attempt < max_retries:
                    print(f"  ⚠️ GitHub API 错误 (状态码: {status})，{retry_delay}秒后重试 (第{attempt}/{max_retries}次)...")
                    time.sleep(retry_delay)
                else:
                    print(f"  ❌ GitHub API 调用多次重试失败 (状态码: {status}): {e}")
            except Exception as e:
                if attempt < max_retries:
                    print(f"  ⚠️ 网络请求异常: {type(e).__name__}，{retry_delay}秒后重试 (第{attempt}/{max_retries}次)...")
                    time.sleep(retry_delay)
                else:
                    print(f"  ❌ 网络请求多次重试失败: {e}")
                    return None
        return None

    def get_repo(self, owner: str, repo_name: str) -> Repository:
        """获取仓库对象（含重试机制）"""
        repo = self._retry_api_call(self.gh.get_repo, f"{owner}/{repo_name}", max_retries=3, retry_delay=10)
        return repo

    def fetch_pulls(
        self,
        repo: Repository,
        state: str = config.PR_STATE,
        sort: str = config.SORT,
        direction: str = config.DIRECTION,
        count: int = config.PR_PER_REPO
    ) -> List[Dict]:
        """获取指定数量的 Pull Request（含重试机制）"""
        pulls: List[Dict] = []
        print(f"开始获取 {repo.full_name} 的 PR，目标数量: {count}...")

        if repo is None:
            print("  仓库对象为空，无法获取 PR")
            return pulls

        pulls_gen = self._retry_api_call(
            repo.get_pulls, max_retries=3, retry_delay=10,
            state=state, sort=sort, direction=direction
        )
        if pulls_gen is None:
            print("  获取 PR 列表失败")
            return pulls

        for i, pr in enumerate(pulls_gen):
            if i >= count:
                break

            pr_data = self._extract_pr_basic(pr)
            pulls.append(pr_data)

            if (i + 1) % 50 == 0:
                print(f"  已获取 {i + 1} 个 PR")

            time.sleep(config.REQUEST_DELAY)

        print(f"完成，共获取 {len(pulls)} 个 PR")
        return pulls

    def _extract_pr_basic(self, pr: PullRequest.PullRequest) -> Dict:
        """提取 PR 基础信息"""
        return {
            "pr_id": pr.number,
            "title": pr.title,
            "body": pr.body if pr.body else "",
            "author": pr.user.login if pr.user else None,
            "created_at": pr.created_at.isoformat() if pr.created_at else None,
            "merged": pr.merged,
            "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
            "closed_at": pr.closed_at.isoformat() if pr.closed_at else None,
            "additions": pr.additions,
            "deletions": pr.deletions,
            "changed_files": pr.changed_files,
            "labels": [label.name for label in pr.labels],
            "html_url": pr.html_url,
        }

    def fetch_reviews_and_comments(
        self,
        repo: Repository,
        pr_number: int
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """获取 PR 的所有 Reviews 和 Issue Comments（含重试机制）"""
        pr = self._retry_api_call(repo.get_pull, pr_number, max_retries=3, retry_delay=10)
        if pr is None:
            return [], [], []

        # 获取 Reviews
        reviews_data: List[Dict] = []
        try:
            reviews_gen = self._retry_api_call(
                lambda: list(pr.get_reviews()),
                max_retries=3, retry_delay=10
            )
            if reviews_gen:
                for review in reviews_gen:
                    review_data = {
                        "review_id": review.id,
                        "reviewer": review.user.login if review.user else None,
                        "state": review.state,
                        "body": review.body if review.body else "",
                        "submitted_at": review.submitted_at.isoformat() if review.submitted_at else None,
                    }
                    reviews_data.append(review_data)
                    time.sleep(config.REQUEST_DELAY)
        except GithubException as e:
            print(f"  获取 Reviews 失败 (PR #{pr_number}): {e}")

        # 获取 Issue Comments（PR 上的讨论评论）
        issue_comments_data: List[Dict] = []
        try:
            issue_comments_gen = self._retry_api_call(
                lambda: list(pr.get_issue_comments()),
                max_retries=3, retry_delay=10
            )
            if issue_comments_gen:
                for comment in issue_comments_gen:
                    comment_data = {
                        "comment_id": comment.id,
                        "user": comment.user.login if comment.user else None,
                        "body": comment.body if comment.body else "",
                        "created_at": comment.created_at.isoformat() if comment.created_at else None,
                    }
                    issue_comments_data.append(comment_data)
                    time.sleep(config.REQUEST_DELAY)
        except GithubException as e:
            print(f"  获取 Issue Comments 失败 (PR #{pr_number}): {e}")

        # 获取 Review Comments（行内代码评论，即针对具体行的评论）
        review_comments_data: List[Dict] = []
        try:
            review_comments_gen = self._retry_api_call(
                lambda: list(pr.get_review_comments()),
                max_retries=3, retry_delay=10
            )
            if review_comments_gen:
                for comment in review_comments_gen:
                    comment_data = {
                        "comment_id": comment.id,
                        "user": comment.user.login if comment.user else None,
                        "body": comment.body if comment.body else "",
                        "path": comment.path if hasattr(comment, 'path') else None,
                        "position": comment.position if hasattr(comment, 'position') else None,
                        "created_at": comment.created_at.isoformat() if comment.created_at else None,
                    }
                    review_comments_data.append(comment_data)
                    time.sleep(config.REQUEST_DELAY)
        except GithubException as e:
            print(f"  获取 Review Comments 失败 (PR #{pr_number}): {e}")

        return reviews_data, issue_comments_data, review_comments_data

    def fetch_files_and_commits(
        self,
        repo: Repository,
        pr_number: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """获取 PR 的修改文件和所有 Commits（含重试机制）"""
        pr = self._retry_api_call(repo.get_pull, pr_number, max_retries=3, retry_delay=10)
        if pr is None:
            return [], [], []

        # 获取修改文件
        files_data: List[Dict] = []
        try:
            files_gen = self._retry_api_call(
                lambda: list(pr.get_files()),
                max_retries=3, retry_delay=10
            )
            if files_gen:
                for file in files_gen:
                    file_data = {
                        "filename": file.filename,
                        "status": file.status,
                        "additions": file.additions,
                        "deletions": file.deletions,
                        "changes": file.changes,
                        "patch": file.patch if hasattr(file, 'patch') else None,
                        "raw_url": file.raw_url,
                        "blob_url": file.blob_url,
                    }
                    # 从 patch 中提取修改的函数名
                    if file_data.get("patch"):
                        file_data["modified_functions"] = self._extract_modified_functions(file_data["patch"], file.filename)
                    else:
                        file_data["modified_functions"] = []
                    files_data.append(file_data)
                    time.sleep(config.REQUEST_DELAY)
        except GithubException as e:
            print(f"  获取 Files 失败 (PR #{pr_number}): {e}")

        # 获取 Commits
        commits_data: List[Dict] = []
        try:
            commits_gen = self._retry_api_call(
                lambda: list(pr.get_commits()),
                max_retries=3, retry_delay=10
            )
            if commits_gen:
                for commit in commits_gen:
                    commit_data = {
                        "sha": commit.sha,
                        "message": commit.commit.message,
                        "author": commit.commit.author.name if commit.commit.author else None,
                        "author_email": commit.commit.author.email if commit.commit.author else None,
                        "date": commit.commit.author.date.isoformat() if commit.commit.author else None,
                        "additions": commit.stats.additions if hasattr(commit, 'stats') else None,
                        "deletions": commit.stats.deletions if hasattr(commit, 'stats') else None,
                    }
                    commits_data.append(commit_data)
                    time.sleep(config.REQUEST_DELAY)
        except GithubException as e:
            print(f"  获取 Commits 失败 (PR #{pr_number}): {e}")

        return files_data, commits_data

    @staticmethod
    def _extract_modified_functions(patch: str, filename: str) -> List[str]:
        """
        从 diff patch 中提取被修改的函数名
        支持多种编程语言（Python、Go、Java、JS/TS、C/C++、Rust 等）
        """
        functions = []
        seen = set()

        # 根据文件扩展名选择正则模式
        ext = os.path.splitext(filename)[1].lower()

        # 通用模式：从 hunk header 中提取函数上下文
        # 例如: @@ -100,7 +100,7 @@ def some_function(param1, param2):
        hunk_pattern = r'@@[^@]*@@\s*(.+)'
        for match in re.finditer(hunk_pattern, patch):
            context = match.group(1).strip()
            # 尝试从上下文中提取函数名
            func_match = re.search(
                r'(?:def\s+|function\s+|func\s+|class\s+)?'
                r'(?:async\s+)?(?:fn\s+)?(\w+)\s*(?:\{|\(|=>)',
                context
            )
            if func_match and func_match.group(1) not in seen:
                name = func_match.group(1)
                # 过滤掉非函数关键词
                if name.lower() not in ('if', 'for', 'while', 'switch', 'return', 'import', 'from'):
                    functions.append(name)
                    seen.add(name)

        # 语言特定模式：从新增/修改的行中提取函数定义
        added_lines = re.findall(r'^\+.*$', patch, re.MULTILINE)

        # Python: def function_name(
        if ext == '.py':
            patterns = [r'^\+\s*def\s+(\w+)\s*\(', r'^\+\s*class\s+(\w+)\s*']
        # Go: func FunctionName(
        elif ext == '.go':
            patterns = [r'^\+\s*func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(']
        # Java: public/private ReturnType functionName(
        elif ext == '.java':
            patterns = [r'^\+\s*(?:public|private|protected|static|\s)*\s+(\w+)\s*\(', r'^\+\s*(?:public|private|protected)?\s*(?:class|interface)\s+(\w+)']
        # JS/TS: function name() or name = () =>
        elif ext in ('.js', '.jsx', '.ts', '.tsx'):
            patterns = [
                r'^\+\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)',
                r'^\+\s*(?:export\s+)?(?:async\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\(',
                r'^\+\s*(?:export\s+)?(?:async\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\w+\s*=>',
            ]
        # C/C++
        elif ext in ('.c', '.cpp', '.cc', '.cxx', '.h', '.hpp'):
            patterns = [r'^\+\s*(?:static\s+)?(?:inline\s+)?(?:\w+\s+)+\*?(\w+)\s*\(']
        # Rust
        elif ext == '.rs':
            patterns = [r'^\+\s*(?:pub\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)']
        else:
            patterns = [
                r'^\+\s*(?:def|function|func|fn)\s+(\w+)',
                r'^\+\s*(?:public|private|protected)?\s*(?:class|interface|struct)\s+(\w+)',
            ]

        for line in added_lines:
            for pattern in patterns:
                match = re.search(pattern, line)
                if match and match.group(1) not in seen:
                    name = match.group(1)
                    functions.append(name)
                    seen.add(name)
                    break

        return list(dict.fromkeys(functions))  # 去重并保持顺序

    def save_raw_data(
        self,
        owner: str,
        repo_name: str,
        pulls: List[Dict],
    ) -> None:
        """保存原始数据到 JSON 文件"""
        output_file = os.path.join(
            config.RAW_DIR,
            f"{owner}_{repo_name}_pulls.json"
        )

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pulls, f, ensure_ascii=False, indent=2)

        print(f"原始数据已保存到: {output_file}")

    def load_raw_data(
        self,
        owner: str,
        repo_name: str
    ) -> Optional[List[Dict]]:
        """从 JSON 文件加载原始数据"""
        input_file = os.path.join(
            config.RAW_DIR,
            f"{owner}_{repo_name}_pulls.json"
        )

        if not os.path.exists(input_file):
            return None

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def has_data(self, owner: str, repo_name: str) -> bool:
        """检查是否已有原始数据"""
        input_file = os.path.join(
            config.RAW_DIR,
            f"{owner}_{repo_name}_pulls.json"
        )
        return os.path.exists(input_file)

def get_all_prs_data() -> List[Dict]:
    """加载所有项目的 PR 数据"""
    import json
    all_pulls = []
    for owner, repo_name, _ in config.TARGET_REPOS:
        input_file = os.path.join(
            config.RAW_DIR,
            f"{owner}_{repo_name}_pulls.json"
        )
        if os.path.exists(input_file):
            with open(input_file, 'r', encoding='utf-8') as f:
                pulls = json.load(f)
                # 添加仓库信息
                for pr in pulls:
                    pr['repo_owner'] = owner
                    pr['repo_name'] = repo_name
                all_pulls.extend(pulls)
            print(f"已加载 {owner}/{repo_name}: {len(pulls)} 个 PR")
    return all_pulls

if __name__ == "__main__":
    # 测试
    fetcher = GithubDataFetcher()
    fetcher._rate_limit_info()