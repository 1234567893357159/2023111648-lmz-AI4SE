# 实验一：代码审查意见挖掘与需求理解

## 实验概述

本实验通过 GitHub API 从 5 个大型开源项目中获取 Pull Request 数据，构建包含丰富特征的数据集，并进行统计分析与可视化。该数据集将作为后续课程实验（Merge Prediction、Review Comment Generation）的数据基础。

## 实验目标

1. 从 GitHub 获取 5 个开源项目的 Pull Request 数据
2. 提取 Code Review、Commit、文件修改等信息
3. 提取特征（AI Reviewer、AI 生成代码检测等）
4. 数据统计分析与可视化

## 项目结构

```
lab1/
├── main.ipynb                 # 主流程 Jupyter Notebook
├── config.py                  # 配置文件（GitHub Token、目标项目、API 参数等）
├── github_api.py              # GitHub API 数据获取模块
├── feature_extractor.py       # 特征提取模块
├── analysis.py                # 统计分析 & 可视化模块
├── 1.py                       # SSL 证书连接测试脚本
├── requirements.txt           # Python 依赖包
├── requst.md                  # 实验要求文档
├── data/
│   ├── raw/                   # 原始 JSON 数据（每个项目一个文件）
│   │   ├── kubernetes_kubernetes_pulls.json
│   │   ├── pytorch_pytorch_pulls.json
│   │   ├── tensorflow_tensorflow_pulls.json
│   │   ├── microsoft_vscode_pulls.json
│   │   └── langchain-ai_langchain_pulls.json
│   └── processed/             # 处理后的 CSV 数据集
│       ├── dataset.csv        # 完整特征数据集（1500 行 × 36 列）
│       └── summary.csv        # 统计摘要
├── figures/                   # 可视化图表
│   ├── 01_merge_vs_non_merge.png
│   ├── 02_comment_distribution.png
│   ├── 03_label_distribution.png
│   ├── 04_reviewer_count_distribution.png
│   ├── 05_pr_length_distribution.png
│   ├── 06_ai_vs_human.png
│   ├── 07_per_repo_comparison.png
│   ├── 08_correlation_heatmap.png
│   └── 09_language_distribution.png
└── report/                    # 实验报告（LaTeX）
    ├── main.tex
    ├── experimentreport.cls
    └── listing_style.tex
```

## 目标项目

| 序号 | 项目 | 描述 |
|------|------|------|
| 1 | kubernetes/kubernetes | 容器编排平台，PR 审查流程严格规范 |
| 2 | pytorch/pytorch | 深度学习框架，有 AI 生成代码和 AI Review |
| 3 | tensorflow/tensorflow | 深度学习框架，大型社区活跃 |
| 4 | microsoft/vscode | 大型 IDE 项目，代码审查规范 |
| 5 | langchain-ai/langchain | LLM 应用框架，与 AI 紧密相关 |

每个项目获取 300 个 PR，共 1500 个 PR。

## 环境要求

- Python 3.10+
- 依赖包见 `requirements.txt`

### 依赖安装

```bash
pip install -r requirements.txt
```

### 依赖列表

| 包名 | 版本 | 用途 |
|------|------|------|
| PyGithub | 2.3.0 | GitHub API 调用 |
| pandas | 2.2.0 | 数据处理 |
| matplotlib | 3.8.0 | 数据可视化 |
| seaborn | 0.13.0 | 统计可视化 |
| requests | 2.31.0 | HTTP 请求 |
| jupyter | 1.0.0 | Notebook 运行环境 |
| tqdm | 4.66.0 | 进度条显示 |

## 快速开始

### 1. 配置 GitHub Token

打开 `config.py`，将 `GITHUB_TOKEN` 替换为你的 GitHub Personal Access Token：

```python
GITHUB_TOKEN = "your_github_token_here"
```

Token 生成方式：GitHub Settings → Developer Settings → Personal Access Tokens → Tokens (classic)，无需任何权限（public repo 只读即可）。

### 2. 运行主流程

在 Jupyter Notebook 中打开 `main.ipynb`，按顺序执行所有 Cell：

- **步骤一**：获取 PR 基础数据（如已缓存则跳过）
- **步骤二**：获取 Code Review 和 Comments（Reviews、Issue Comments、Review Comments、Files、Commits）
- **步骤三**：数据概览（展示数据样例）
- **步骤四**：特征提取（构建 36 维特征数据集）
- **步骤五**：统计分析与可视化（生成 9 张图表）

### 3. 独立运行分析

如果已有处理好的数据集，可以直接运行分析：

```python
from analysis import load_dataset, DataAnalyzer

df = load_dataset()
analyzer = DataAnalyzer(df)
analyzer.run_all_analyses()
```

## 模块说明

### config.py — 配置文件

| 配置项 | 说明 |
|--------|------|
| `GITHUB_TOKEN` | GitHub API 访问令牌 |
| `REQUEST_DELAY` | API 请求间隔（秒），防止限流 |
| `TARGET_REPOS` | 目标项目列表（owner, repo_name, 描述） |
| `PR_PER_REPO` | 每个项目获取的 PR 数量（默认 300） |
| `AI_REVIEWER_KEYWORDS` | AI/Bot Reviewer 检测关键词 |
| `BOT_AUTHOR_KEYWORDS` | Bot 作者排除关键词 |
| `AI_KEYWORDS` | AI 生成代码检测关键词 |

### github_api.py — 数据获取模块

`GithubDataFetcher` 类实现了以下功能：

- `fetch_pulls()` — 获取指定数量的 PR 列表
- `fetch_reviews_and_comments()` — 获取 PR 的 Reviews、Issue Comments 和行内 Review Comments
- `fetch_files_and_commits()` — 获取 PR 的修改文件和 Commits，并提取修改函数名
- `save_raw_data()` / `load_raw_data()` — 原始数据的保存与加载（支持断点续传）
- `_retry_api_call()` — 通用 API 调用重试机制（自动处理限流和网络错误）

### feature_extractor.py — 特征提取模块

`FeatureExtractor` 类提取 36 个特征，分为以下几类：

**基础特征（16 个）**：
`pr_id`, `repo_owner`, `repo_name`, `title`, `author`, `created_at`, `pr_length`, `title_length`, `files_changed`, `additions`, `deletions`, `is_merged`, `label_count`, `labels`, `modified_function_count`, `modified_functions`

**Review 特征（12 个）**：
`reviewer_count`, `reviewers`, `review_count`, `review_comment_count`, `issue_comment_count`, `inline_comment_count`, `total_comment_count`, `approved_count`, `changes_requested_count`, `commented_count`, `final_review_decision`, `has_review`

**AI 检测特征（5 个）**：
`has_ai_reviewer`, `ai_reviewers`, `ai_reviewer_count`, `has_ai_generated_code`, `ai_code_indicators`

**语言特征（3 个）**：
`primary_language`, `total_languages`, `languages`

### analysis.py — 统计分析模块

`DataAnalyzer` 类实现 9 项分析：

| 序号 | 分析内容 | 图表文件 |
|------|----------|----------|
| 1 | Merge vs Non-Merge 数量统计（柱状图 + 饼图） | `01_merge_vs_non_merge.png` |
| 2 | Review Comment 数量分布（直方图） | `02_comment_distribution.png` |
| 3 | Label 分布（Top 15 水平柱状图） | `03_label_distribution.png` |
| 4 | Reviewer 数量分布（直方图） | `04_reviewer_count_distribution.png` |
| 5 | PR 长度分布（原始尺度 + 对数尺度） | `05_pr_length_distribution.png` |
| 6 | AI vs Human 参与情况（AI Reviewer + AI 生成代码饼图） | `06_ai_vs_human.png` |
| 7 | 各仓库关键指标对比（Merge Rate、Reviewer、AI 指标） | `07_per_repo_comparison.png` |
| 8 | 特征相关性热力图 | `08_correlation_heatmap.png` |
| 9 | 编程语言分布（按 PR 出现数 + 按主语言） | `09_language_distribution.png` |

## 数据集概览

- **数据量**：1500 个 PR，36 个特征
- **数据来源**：5 个 GitHub 开源项目，每个 300 个 PR
- **Merge Rate**：约 36.73%
- **AI Reviewer 参与率**：约 74.1%（多数为 CI Bot）
- **AI 生成代码占比**：约 22.9%

## 技术要点

### AI 生成代码检测

采用三阶段检测策略：

1. **Bot 作者排除**：如果 PR 作者为已知 Bot（如 dependabot、renovate），直接排除
2. **关键词匹配**：在 Commit Message、PR Body、Title、Labels 中匹配 AI 工具关键词（Copilot、ChatGPT、Claude、Cursor、DeepSeek 等）
3. **启发式规则**：单文件大新增 + 短 commit 组合、PR Body 高度模板化等特征

### AI Reviewer 检测

通过 Reviewer 用户名中包含的关键词（bot、code-review、copilot、sonar、codacy 等）来判断是否为 AI/Bot Reviewer。

## 注意事项

1. GitHub API 有频率限制（未认证 60 次/小时，已认证 5000 次/小时），建议配置 Token
2. 原始数据已缓存在 `data/raw/` 目录，如需重新获取请删除对应 JSON 文件
3. 处理过程中遇到 API 错误会自动重试（最多 3 次）
4. 可视化图表使用 `matplotlib.use('Agg')` 后端，支持非交互式环境