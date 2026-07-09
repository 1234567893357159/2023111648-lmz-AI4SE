"""
实验三 配置文件
数据集划分、模型路径等配置
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ========== 基础路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAB1_BASE_DIR = os.path.join(os.path.dirname(BASE_DIR), "lab1")
LAB2_BASE_DIR = os.path.join(os.path.dirname(BASE_DIR), "lab2")

# ========== Lab1 数据路径 ==========
LAB1_DATA_DIR = os.path.join(LAB1_BASE_DIR, "data")
LAB1_RAW_DIR = os.path.join(LAB1_DATA_DIR, "raw")

# ========== Lab2 数据路径 ==========
LAB2_DATA_DIR = os.path.join(LAB2_BASE_DIR, "data")
LAB2_CODE_DIR = os.path.join(LAB2_DATA_DIR, "code")
LAB2_RAW_DIR = os.path.join(LAB2_DATA_DIR, "raw")
LAB2_HUMAN_RAW_DIR = os.path.join(LAB2_RAW_DIR, "human")

# ========== 输出路径（本实验）==========
OUTPUT_DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_JSON_PATH = os.path.join(OUTPUT_DATA_DIR, "train.json")
VAL_JSON_PATH = os.path.join(OUTPUT_DATA_DIR, "val.json")
TEST_JSON_PATH = os.path.join(OUTPUT_DATA_DIR, "test.json")

# 创建输出目录
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

# ========== 目标项目列表（与 lab2 保持一致）==========
TARGET_REPOS = [
    ("kubernetes", "kubernetes"),
    ("pytorch", "pytorch"),
    ("tensorflow", "tensorflow"),
    ("microsoft", "vscode"),
    ("langchain-ai", "langchain"),
]

# ========== 数据集划分比例 ==========
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ========== 随机种子（保证可复现）==========
RANDOM_SEED = 42

# ========== Code2Vec 向量化配置 ==========
VECTORS_DIR = os.path.join(OUTPUT_DATA_DIR, "vectors")
VOCAB_PATH = os.path.join(VECTORS_DIR, "vocab.json")
TRAIN_VECTORS_PATH = os.path.join(VECTORS_DIR, "train_vectors.pt")
VAL_VECTORS_PATH = os.path.join(VECTORS_DIR, "val_vectors.pt")
TEST_VECTORS_PATH = os.path.join(VECTORS_DIR, "test_vectors.pt")

os.makedirs(VECTORS_DIR, exist_ok=True)

# AST 路径缓存目录（每个 PR 单独保存，避免重复解析）
AST_CACHE_DIR = os.path.join(VECTORS_DIR, "ast_cache")
os.makedirs(AST_CACHE_DIR, exist_ok=True)

# 向量化参数
MAX_PATH_LENGTH = 8           # 每条 AST 路径最大节点数
MAX_PATHS_PER_FILE = 1000      # 每个文件最多提取的路径数
MAX_LEAVES_PER_FILE = 500      # 每个文件最多收集的叶子节点数
CODE_VECTOR_DIM = 128         # 输出代码向量维度
TOKEN_EMBED_DIM = 64          # token 嵌入维度
PATH_EMBED_DIM = 64           # 路径嵌入维度
CONTEXT_DIM = 128             # 上下文向量维度
MIN_PATH_FREQ = 5             # 词汇表最小频率

# 语言扩展名映射
SUPPORTED_EXTENSIONS = {
    ".go": "Go", ".py": "Python", ".pyi": "Python",
    ".js": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".java": "Java", ".c": "C", ".h": "C",
    ".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".hpp": "C++",
}

# tree-sitter 语言名称映射
TREE_SITTER_LANG_MAP = {
    "Go": "go", "Python": "python",
    "JavaScript": "javascript", "TypeScript": "typescript",
    "Java": "java", "C": "c", "C++": "cpp",
}

# ========== 分类器训练配置 ==========
RESULTS_DIR = os.path.join(OUTPUT_DATA_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "image")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 随机森林参数
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_SPLIT = 5
RF_CLASS_WEIGHT = "balanced"

# SVM 参数
SVM_C = 1.0
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"
SVM_CLASS_WEIGHT = "balanced"

# MLP 参数
MLP_HIDDEN_LAYERS = (128, 64)
MLP_MAX_ITER = 500
MLP_LEARNING_RATE = 0.001
MLP_BATCH_SIZE = 32
MLP_EARLY_STOPPING = True
MLP_VALIDATION_FRACTION = 0.1

# 评估指标列表
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# ========== CodeBERT 微调配置 ==========
CODEBERT_DATA_DIR = os.path.join(OUTPUT_DATA_DIR, "codebert")
CODEBERT_TRAIN_PATH = os.path.join(CODEBERT_DATA_DIR, "train.json")
CODEBERT_VAL_PATH = os.path.join(CODEBERT_DATA_DIR, "val.json")
CODEBERT_TEST_PATH = os.path.join(CODEBERT_DATA_DIR, "test.json")
CODEBERT_MODEL_DIR = os.path.join(OUTPUT_DATA_DIR, "codebert_models", "codebert_merge")

os.makedirs(CODEBERT_DATA_DIR, exist_ok=True)
os.makedirs(CODEBERT_MODEL_DIR, exist_ok=True)

CODEBERT_MODEL_NAME = "microsoft/codebert-base"
CODEBERT_MAX_LENGTH = 512
CODEBERT_BATCH_SIZE = 8
CODEBERT_LEARNING_RATE = 2e-5
CODEBERT_EPOCHS = 3
CODEBERT_WEIGHT_DECAY = 0.01

# ========== CodeT5 代码审查意见生成配置 ==========
CODEREVIEW_DATA_DIR = os.path.join(OUTPUT_DATA_DIR, "codereview")
CODEREVIEW_TRAIN_PATH = os.path.join(CODEREVIEW_DATA_DIR, "train.json")
CODEREVIEW_VAL_PATH = os.path.join(CODEREVIEW_DATA_DIR, "val.json")
CODEREVIEW_TEST_PATH = os.path.join(CODEREVIEW_DATA_DIR, "test.json")
CODET5_MODEL_DIR = os.path.join(OUTPUT_DATA_DIR, "codebert_models", "codet5_review")

os.makedirs(CODEREVIEW_DATA_DIR, exist_ok=True)
os.makedirs(CODET5_MODEL_DIR, exist_ok=True)

CODET5_MODEL_NAME = "Salesforce/codet5-base"
CODET5_MAX_INPUT_LENGTH = 1024
CODET5_MAX_TARGET_LENGTH = 256
CODET5_BATCH_SIZE = 1
CODET5_GRADIENT_ACCUMULATION_STEPS = 8
CODET5_LEARNING_RATE = 3e-5
CODET5_EPOCHS = 8
CODET5_WEIGHT_DECAY = 0.01
CODET5_WARMUP_STEPS = 100