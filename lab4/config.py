"""
实验四 配置文件
数据准备、模型调用等配置
"""

import os

# ========== 基础路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAB1_BASE_DIR = os.path.join(os.path.dirname(BASE_DIR), "lab1")

# ========== Lab1 原始数据路径 ==========
LAB1_RAW_DIR = os.path.join(LAB1_BASE_DIR, "data", "raw")

# ========== Lab4 输出路径 ==========
LAB4_DATA_DIR = os.path.join(BASE_DIR, "data")
LAB4_FIGURES_DIR = os.path.join(BASE_DIR, "figures")
LAB4_RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(LAB4_DATA_DIR, exist_ok=True)
os.makedirs(LAB4_FIGURES_DIR, exist_ok=True)
os.makedirs(LAB4_RESULTS_DIR, exist_ok=True)

# ========== 目标仓库（与 lab1 一致）==========
TARGET_REPOS = [
    ("kubernetes", "kubernetes"),
    ("pytorch", "pytorch"),
    ("tensorflow", "tensorflow"),
    ("microsoft", "vscode"),
    ("langchain-ai", "langchain"),
]

# ========== 数据挑选参数 ==========
RANDOM_SEED = 42        # 随机种子，保证可复现
SELECT_COUNT = 100      # 随机挑选的 PR 数量

# ========== 输出文件路径 ==========
SELECTED_PRS_PATH = os.path.join(LAB4_DATA_DIR, "selected_prs.json")
SUMMARY_PATH = os.path.join(LAB4_DATA_DIR, "summary.json")
CONTEXTS_PATH = os.path.join(LAB4_DATA_DIR, "contexts.json")
RESULTS_PATH = os.path.join(LAB4_DATA_DIR, "results.json")
RESULTS_DIR = LAB4_RESULTS_DIR

# ========== Ollama 本地模型配置 ==========
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_TEMPERATURE = 0.0
OLLAMA_MAX_TOKENS = 512
MAX_RETRIES = 3
REQUEST_INTERVAL = 0.1
REQUEST_TIMEOUT = 300
MAX_PROMPT_CHARS = 25000