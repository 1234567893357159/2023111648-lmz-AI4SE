"""
代码提取模块
从 GitHub 下载每个 PR 修改前后的完整源代码文件
"""

import json
import os
import re
import time
import certifi
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

import config

requests.utils.DEFAULT_CA_BUNDLE_PATH = certifi.where()


class CodeExtractor:
    """代码提取器：下载 PR 修改前后的源文件"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {config.GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        })

        # 代理配置：优先使用 config 中的配置，否则使用环境变量
        proxies = {}
        if config.HTTP_PROXY:
            proxies["http"] = config.HTTP_PROXY
        if config.HTTPS_PROXY:
            proxies["https"] = config.HTTPS_PROXY
        if not proxies:
            proxies = None
        self.session.proxies.update(proxies or {})

        # 重试策略：遇到超时/连接错误自动重试 3 次
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.parent_sha_cache = {}
        self.stats = {"downloaded": 0, "skipped": 0, "failed": 0, "no_before": 0}

    def _get_filename(self, filepath: str) -> str:
        """将文件路径转为安全的文件名（/ 替换为 _）"""
        return filepath.replace("/", "_").replace("\\", "_")

    def _get_language(self, filepath: str) -> Optional[str]:
        """根据扩展名判断是否支持的语言，返回语言名或 None"""
        ext = os.path.splitext(filepath)[1].lower()
        if not ext and filepath.lower() == "dockerfile":
            return None
        return config.SUPPORTED_EXTENSIONS.get(ext)

    def _extract_sha_from_url(self, raw_url: str) -> Optional[str]:
        """从 raw_url 中提取 commit SHA"""
        match = re.search(r"/raw/([a-f0-9]{40})/", raw_url)
        return match.group(1) if match else None

    def _get_parent_sha(self, owner: str, repo_name: str, commit_sha: str) -> Optional[str]:
        """通过 GitHub API 获取 commit 的 parent SHA"""
        if commit_sha in self.parent_sha_cache:
            return self.parent_sha_cache[commit_sha]

        url = f"https://api.github.com/repos/{owner}/{repo_name}/commits/{commit_sha}"
        try:
            resp = self.session.get(url, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                parents = data.get("parents", [])
                if parents:
                    parent_sha = parents[0]["sha"]
                    self.parent_sha_cache[commit_sha] = parent_sha
                    return parent_sha
            elif resp.status_code == 403:
                print(f"    ⚠️ API 限流，等待 60 秒...")
                time.sleep(60)
                return self._get_parent_sha(owner, repo_name, commit_sha)
            else:
                print(f"    ⚠️ 获取 parent commit 失败: HTTP {resp.status_code}")
        except Exception as e:
            print(f"    ⚠️ 网络错误: {e}")
        return None

    def _download_raw_file(self, url: str, notfound_marker: str = None) -> Optional[str]:
        """下载 GitHub raw 文件内容，支持 github.com/raw 和 raw.githubusercontent.com
        
        Args:
            url: 下载链接
            notfound_marker: 如果提供，404 时会创建这个标记文件，后续不再重试
        """
        # 将 github.com/.../raw/... 格式转为 raw.githubusercontent.com 格式
        # 后者在国内环境下更容易通过代理访问
        url = re.sub(
            r"https://github\.com/([^/]+)/([^/]+)/raw/([a-f0-9]+)/(.+)",
            r"https://raw.githubusercontent.com/\1/\2/\3/\4",
            url
        )
        try:
            resp = self.session.get(url, timeout=60)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 404:
                if notfound_marker:
                    os.makedirs(os.path.dirname(notfound_marker), exist_ok=True)
                    with open(notfound_marker, "w", encoding="utf-8") as f:
                        f.write("")
                print(f"      ⚠️ 404 不存在，已标记跳过: {url[:80]}...")
            elif resp.status_code == 403:
                print(f"      ⚠️ 下载限流，等待 60 秒...")
                time.sleep(60)
                return self._download_raw_file(url, notfound_marker)
            else:
                print(f"      ⚠️ 下载失败: HTTP {resp.status_code} - {url[:80]}...")
        except requests.exceptions.Timeout:
            print(f"      ⚠️ 下载超时 (60s): {url[:80]}...")
        except requests.exceptions.ConnectionError as e:
            print(f"      ⚠️ 连接失败: {e}")
        except Exception as e:
            print(f"      ⚠️ 下载异常: {type(e).__name__}: {e}")
        return None

    def _get_before_url(self, owner: str, repo_name: str, raw_url: str, filename: str) -> Optional[str]:
        """构造 before 版本的 raw URL"""
        commit_sha = self._extract_sha_from_url(raw_url)
        if not commit_sha:
            return None

        parent_sha = self._get_parent_sha(owner, repo_name, commit_sha)
        if not parent_sha:
            return None

        return f"https://raw.githubusercontent.com/{owner}/{repo_name}/{parent_sha}/{filename}"

    def extract_pr_code(self, owner: str, repo_name: str, pr_data: Dict, force_check: bool = False) -> Dict:
        """提取单个 PR 的代码"""
        pr_id = pr_data["pr_id"]
        repo_key = f"{owner}_{repo_name}"
        pr_dir = os.path.join(config.HUMAN_CODE_DIR, repo_key, str(pr_id))
        before_dir = os.path.join(pr_dir, "before")
        after_dir = os.path.join(pr_dir, "after")
        os.makedirs(before_dir, exist_ok=True)
        os.makedirs(after_dir, exist_ok=True)

        files = pr_data.get("files", [])
        extracted = []

        for file_info in files:
            filename = file_info.get("filename", "")
            status = file_info.get("status", "modified")
            raw_url = file_info.get("raw_url", "")

            lang = self._get_language(filename)
            if lang is None:
                continue

            safe_name = self._get_filename(filename)
            after_path = os.path.join(after_dir, safe_name)
            before_path = os.path.join(before_dir, safe_name)

            file_result = {
                "filename": filename,
                "safe_name": safe_name,
                "language": lang,
                "status": status,
            }

            NOTFOUND_EXT = ".notfound"

            # 快速跳过：非强制检查时，只要文件存在就直接跳过，不做后续操作
            if status != "removed":
                notfound = after_path + NOTFOUND_EXT
                if not force_check:
                    if os.path.exists(notfound) or os.path.exists(after_path):
                        self.stats["skipped"] += 1
                        file_result["after"] = after_path if os.path.exists(after_path) else None
                        extracted.append(file_result)
                        continue
                if os.path.exists(notfound):
                    self.stats["skipped"] += 1
                elif os.path.exists(after_path):
                    self.stats["skipped"] += 1
                else:
                    content = self._download_raw_file(raw_url, notfound_marker=notfound)
                    if content:
                        with open(after_path, "w", encoding="utf-8") as f:
                            f.write(content)
                        self.stats["downloaded"] += 1
                        file_result["after"] = after_path
                    else:
                        self.stats["failed"] += 1
                        file_result["after"] = None

            if status != "added":
                notfound = before_path + NOTFOUND_EXT
                if not force_check:
                    if os.path.exists(notfound) or os.path.exists(before_path):
                        self.stats["skipped"] += 1
                        file_result["before"] = before_path if os.path.exists(before_path) else None
                        extracted.append(file_result)
                        continue
                if os.path.exists(notfound):
                    self.stats["skipped"] += 1
                elif os.path.exists(before_path):
                    self.stats["skipped"] += 1
                else:
                    before_url = self._get_before_url(owner, repo_name, raw_url, filename)
                    if before_url:
                        content = self._download_raw_file(before_url, notfound_marker=notfound)
                        if content:
                            with open(before_path, "w", encoding="utf-8") as f:
                                f.write(content)
                            self.stats["downloaded"] += 1
                            file_result["before"] = before_path
                        else:
                            self.stats["failed"] += 1
                            file_result["before"] = None
                    else:
                        self.stats["no_before"] += 1
                        file_result["before"] = None

            extracted.append(file_result)

        return {"pr_id": pr_id, "files": extracted}

    def extract_repo(self, owner: str, repo_name: str, force_check: bool = False) -> List[Dict]:
        """提取单个仓库所有 PR 的代码"""
        repo_key = f"{owner}_{repo_name}"
        human_file = os.path.join(config.HUMAN_DIR, f"{repo_key}_pulls.json")

        if not os.path.exists(human_file):
            print(f"  未找到 {repo_key} 的数据，跳过")
            return []

        with open(human_file, "r", encoding="utf-8") as f:
            prs = json.load(f)

        print(f"  共 {len(prs)} 个 PR，开始提取代码...")

        results = []
        pbar = tqdm(prs, desc=f"  {owner}/{repo_name}", unit="pr",
                    ncols=100, ascii=True)
        for pr in pbar:
            result = self.extract_pr_code(owner, repo_name, pr, force_check=force_check)
            results.append(result)
            pbar.set_postfix({
                "下载": self.stats["downloaded"],
                "跳过": self.stats["skipped"],
                "失败": self.stats["failed"],
            })
            time.sleep(config.REQUEST_DELAY)
        pbar.close()

        return results

    def run_all(self, force_check: bool = False):
        """运行所有仓库的代码提取
        
        Args:
            force_check: False（默认）啥都不做直接返回；True 遍历检查并下载缺失文件
        """
        if not force_check:
            print("=" * 60)
            print("快速模式：跳过检查，默认所有文件已下载完成")
            print("=" * 60)
            return {}
        
        print("=" * 60)
        print("开始提取 PR 代码...")
        print("=" * 60)

        all_results = {}
        for owner, repo_name in config.TARGET_REPOS:
            print(f"\n处理 {owner}/{repo_name}...")
            results = self.extract_repo(owner, repo_name, force_check=force_check)
            all_results[f"{owner}_{repo_name}"] = results

        print(f"\n{'=' * 60}")
        print(f"代码提取完成!")
        print(f"  下载成功: {self.stats['downloaded']}")
        print(f"  跳过(已存在): {self.stats['skipped']}")
        print(f"  下载失败: {self.stats['failed']}")
        print(f"  无 before: {self.stats['no_before']}")
        print(f"{'=' * 60}")

        return all_results


if __name__ == "__main__":
    extractor = CodeExtractor()
    extractor.run_all()