"""
统计训练数据中 input 和 target 的 token 长度分布
帮助选择合适的 MAX_INPUT_LENGTH 和 MAX_TARGET_LENGTH
"""

import json
import numpy as np
from transformers import AutoTokenizer
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config


def analyze_length_distribution():
    print("=" * 70)
    print("统计序列长度分布：input (diff + PR info) 和 target (review comments)")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(config.CODET5_MODEL_NAME, clean_up_tokenization_spaces=False)

    all_input_lengths = []
    all_target_lengths = []
    total_samples = 0

    # 统计所有数据集（train+val+test）
    for split_name, path in [
        ("训练集", config.CODEREVIEW_TRAIN_PATH),
        ("验证集", config.CODEREVIEW_VAL_PATH),
        ("测试集", config.CODEREVIEW_TEST_PATH),
    ]:
        print(f"\n正在处理 {split_name}...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            input_len = len(tokenizer.encode(item["input"]))
            target_len = len(tokenizer.encode(item["target"]))
            all_input_lengths.append(input_len)
            all_target_lengths.append(target_len)
            total_samples += 1

    all_input_lengths = np.array(all_input_lengths)
    all_target_lengths = np.array(all_target_lengths)

    print(f"\n总计样本数: {total_samples}")
    print("-" * 70)

    # 输入长度统计
    print("\n📊 输入长度 (input = title + body + diff + patches) 分布：")
    print(f"  最小值:  {all_input_lengths.min():>6d} tokens")
    print(f"  平均值:  {all_input_lengths.mean():>6.1f} tokens")
    print(f"  中位数:  {np.median(all_input_lengths):>6.0f} tokens")
    print(f"  90% 分位: {np.percentile(all_input_lengths, 90):>6.0f} tokens")
    print(f"  95% 分位: {np.percentile(all_input_lengths, 95):>6.0f} tokens")
    print(f"  99% 分位: {np.percentile(all_input_lengths, 99):>6.0f} tokens")
    print(f"  最大值:  {all_input_lengths.max():>6d} tokens")

    print("\n📊 目标长度 (target = review comments) 分布：")
    print(f"  最小值:  {all_target_lengths.min():>6d} tokens")
    print(f"  平均值:  {all_target_lengths.mean():>6.1f} tokens")
    print(f"  中位数:  {np.median(all_target_lengths):>6.0f} tokens")
    print(f"  90% 分位: {np.percentile(all_target_lengths, 90):>6.0f} tokens")
    print(f"  95% 分位: {np.percentile(all_target_lengths, 95):>6.0f} tokens")
    print(f"  99% 分位: {np.percentile(all_target_lengths, 99):>6.0f} tokens")
    print(f"  最大值:  {all_target_lengths.max():>6d} tokens")

    # 计算不同截断长度下的覆盖率
    print("\n📈 不同输入长度截断的覆盖率：")
    for length in [256, 384, 512, 640, 768, 1024]:
        coverage = (all_input_lengths <= length).sum() / len(all_input_lengths) * 100
        print(f"  截断到 {length:>4d} tokens: {coverage:>5.1f}% 的样本能完整放下")

    print("\n📈 不同目标长度截断的覆盖率：")
    for length in [64, 128, 160, 200, 256, 512]:
        coverage = (all_target_lengths <= length).sum() / len(all_target_lengths) * 100
        print(f"  截断到 {length:>4d} tokens: {coverage:>5.1f}% 的样本能完整放下")

    print("\n" + "=" * 70)
    print("建议：选择能覆盖 90%-95% 样本的截断长度")
    print("=" * 70)


if __name__ == "__main__":
    analyze_length_distribution()