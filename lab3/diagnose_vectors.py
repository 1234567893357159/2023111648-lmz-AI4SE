"""
向量质量诊断脚本
评估 Code2Vec 生成的向量是否合理：
  1. 路径数统计 - 每个 PR 的 before/after 目录实际抽出多少条路径
  2. 路径饱和度 - 是否大量 PR 都达到 200 上限
  3. 向量多样性 - 不同 PR 的向量是否真的不同
  4. before/after 区分度 - 同一 PR 的 before 和 after 向量是否不同
  5. 按仓库/标签分组分析
"""

import json
import os
import sys
import numpy as np
from collections import Counter, defaultdict

import config

# ========== 第 1 部分：路径数统计 ==========

def analyze_path_counts():
    """分析 AST 缓存中每个 PR 的路径数分布"""
    print("=" * 70)
    print("【第 1 部分】路径数统计")
    print("=" * 70)

    cache_dir = config.AST_CACHE_DIR
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
    n_total = len(cache_files)
    print(f"AST 缓存文件总数: {n_total}")

    if n_total == 0:
        print("❌ 没有缓存文件，请先运行步骤二生成 AST 缓存")
        return None

    before_counts = []
    after_counts = []

    for fname in cache_files:
        fpath = os.path.join(cache_dir, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 兼容新旧缓存格式
        if "before_leaves" in data:
            before_counts.append(len(data['before_leaves']))
            after_counts.append(len(data['after_leaves']))
        elif "before_paths" in data:
            before_counts.append(len(data['before_paths']))
            after_counts.append(len(data['after_paths']))

    before_arr = np.array(before_counts)
    after_arr = np.array(after_counts)

    # 统计
    print(f"\n{'指标':<20} {'Before':>15} {'After':>15}")
    print("-" * 50)
    for label, b_val, a_val in [
        ("最小值", before_arr.min(), after_arr.min()),
        ("最大值", before_arr.max(), after_arr.max()),
        ("平均值", before_arr.mean(), after_arr.mean()),
        ("中位数", np.median(before_arr), np.median(after_arr)),
        ("标准差", before_arr.std(), after_arr.std()),
    ]:
        print(f"{label:<20} {b_val:>15.1f} {a_val:>15.1f}")

    # 饱和度分析
    print(f"\n--- 路径数饱和度分析 (上限 = {config.MAX_PATHS_PER_FILE}) ---")
    for label, arr in [("Before", before_arr), ("After", after_arr)]:
        at_limit = (arr >= config.MAX_PATHS_PER_FILE).sum()
        near_limit = ((arr >= config.MAX_PATHS_PER_FILE * 0.9) & (arr < config.MAX_PATHS_PER_FILE)).sum()
        half_limit = ((arr >= config.MAX_PATHS_PER_FILE * 0.5) & (arr < config.MAX_PATHS_PER_FILE * 0.9)).sum()
        below_half = (arr < config.MAX_PATHS_PER_FILE * 0.5).sum()

        print(f"\n  {label}:")
        print(f"    达到上限 (==200):  {at_limit:>5} ({at_limit/n_total*100:.1f}%)")
        print(f"    接近上限 (180-199): {near_limit:>5} ({near_limit/n_total*100:.1f}%)")
        print(f"    中等 (100-179):     {half_limit:>5} ({half_limit/n_total*100:.1f}%)")
        print(f"    较少 (<100):        {below_half:>5} ({below_half/n_total*100:.1f}%)")

    # 分布直方图 - 扩展到 2000+
    print(f"\n--- 路径数分布直方图 ---")
    print(f"{'区间':<15} {'Before':>8} {'After':>8}  {'可视化'}")
    bins = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 400),
            (400, 500), (500, 700), (700, 1000), (1000, 1500), (1500, 2000), (2000, 9999)]
    max_count = max(
        max(((before_arr >= lo) & (before_arr < hi)).sum() for lo, hi in bins),
        max(((after_arr >= lo) & (after_arr < hi)).sum() for lo, hi in bins),
    )
    for lo, hi in bins:
        b_cnt = ((before_arr >= lo) & (before_arr < hi)).sum()
        a_cnt = ((after_arr >= lo) & (after_arr < hi)).sum()
        bar_len = max(b_cnt, a_cnt) * 40 // max(1, max_count)
        if hi < 9999:
            label = f"{lo}-{hi}"
        else:
            label = f"{lo}+"
        bar = "#" * bar_len
        print(f"{label:<15} {b_cnt:>8} {a_cnt:>8}  {bar}")

    return before_arr, after_arr


# ========== 第 2 部分：加载数据集和标签 ==========

def load_dataset_labels():
    """加载训练集/测试集中的 PR ID 和 label"""
    pr_labels = {}
    for json_path in [config.TRAIN_JSON_PATH, config.VAL_JSON_PATH, config.TEST_JSON_PATH]:
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                pr_labels[item['pr_id']] = {
                    'label': item['label'],
                    'repo': item['repo'],
                    'merged': item.get('merged', False),
                }
    return pr_labels


# ========== 第 3 部分：向量多样性分析 ==========

def analyze_vector_diversity():
    """分析已生成向量的多样性"""
    print("\n" + "=" * 70)
    print("【第 2 部分】向量多样性分析")
    print("=" * 70)

    vectors_path = config.TRAIN_VECTORS_PATH
    if not os.path.exists(vectors_path):
        print("❌ 向量文件还不存在，请先运行步骤二生成向量。")
        print(f"   预期路径: {vectors_path}")
        print("\n   跳过向量分析，仅展示路径统计...")
        return

    import torch
    data = torch.load(vectors_path, weights_only=False)
    before_vecs = data['before_vectors'].numpy()
    after_vecs = data['after_vectors'].numpy()
    labels = data['labels'].numpy()
    pr_ids = data['pr_ids'].numpy()

    n = len(pr_ids)
    dim = before_vecs.shape[1]
    print(f"样本数: {n}, 向量维度: {dim}")
    print(f"正样本(已合并): {labels.sum()}, 负样本(未合并): {(labels == 0).sum()}")

    # 3.1 before 向量自身多样性
    print("\n--- 3.1 Before 向量间的余弦相似度 ---")
    analyze_pairwise_similarity(before_vecs, labels, "Before")

    # 3.2 after 向量自身多样性
    print("\n--- 3.2 After 向量间的余弦相似度 ---")
    analyze_pairwise_similarity(after_vecs, labels, "After")

    # 3.3 before vs after 区分度
    print("\n--- 3.3 同一 PR 的 Before vs After 区分度 ---")
    analyze_before_after_diff(before_vecs, after_vecs, labels)

    # 3.4 拼接向量多样性
    print("\n--- 3.4 拼接向量 [before, after] 的多样性 ---")
    concat_vecs = np.concatenate([before_vecs, after_vecs], axis=1)
    analyze_pairwise_similarity(concat_vecs, labels, "Concat")

    # 3.5 按仓库分组分析
    print("\n--- 3.5 按仓库分组 ---")
    pr_labels = load_dataset_labels()
    analyze_by_repo(before_vecs, after_vecs, pr_ids, pr_labels)


def analyze_pairwise_similarity(vectors, labels, name):
    """计算向量间的余弦相似度分布"""
    # 归一化
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = vectors / norms

    # 采样计算（避免 O(n^2)）
    n = len(vectors)
    sample_size = min(n, 500)
    if n > sample_size:
        indices = np.random.choice(n, sample_size, replace=False)
        normalized = normalized[indices]
        labels = labels[indices]

    # 余弦相似度矩阵
    sim_matrix = normalized @ normalized.T

    # 提取非对角线元素
    mask = ~np.eye(len(normalized), dtype=bool)
    all_sims = sim_matrix[mask]

    # 同类和异类相似度
    same_class_mask = (labels[:, None] == labels[None, :]) & mask
    diff_class_mask = (labels[:, None] != labels[None, :]) & mask

    same_sims = sim_matrix[same_class_mask] if same_class_mask.any() else np.array([0])
    diff_sims = sim_matrix[diff_class_mask] if diff_class_mask.any() else np.array([0])

    print(f"  所有样本对余弦相似度: 均值={all_sims.mean():.4f}, 中位数={np.median(all_sims):.4f}, "
          f"标准差={all_sims.std():.4f}")
    print(f"  同类(同标签)样本对:    均值={same_sims.mean():.4f}, 中位数={np.median(same_sims):.4f}")
    print(f"  异类(不同标签)样本对:  均值={diff_sims.mean():.4f}, 中位数={np.median(diff_sims):.4f}")

    # 判别力指标：异类 vs 同类的相似度差异
    if same_sims.any() and diff_sims.any():
        separation = diff_sims.mean() - same_sims.mean()
        if separation < 0:
            print(f"  ✅ 同类更相似，异类更不同 (差异={abs(separation):.4f})，向量有区分力")
        else:
            print(f"  ⚠️ 异类比同类更相似 (差异={separation:.4f})，向量区分力差！")

    # 相似度极高 (>0.95) 的比例
    very_high = (all_sims > 0.95).mean()
    high = ((all_sims > 0.8) & (all_sims <= 0.95)).mean()
    mid = ((all_sims > 0.5) & (all_sims <= 0.8)).mean()
    low = (all_sims <= 0.5).mean()

    print(f"  相似度分布: >0.95={very_high:.1%}, 0.8-0.95={high:.1%}, "
          f"0.5-0.8={mid:.1%}, <=0.5={low:.1%}")

    if very_high > 0.5:
        print(f"  ⚠️ {very_high:.1%} 的样本对相似度 > 0.95，向量可能塌缩（太相似）！")
    elif very_high > 0.2:
        print(f"  ⚡ {very_high:.1%} 的样本对相似度 > 0.95，偏高但尚可接受")
    else:
        print(f"  ✅ 相似度 > 0.95 的比例较低 ({very_high:.1%})，向量多样性好")


def analyze_before_after_diff(before_vecs, after_vecs, labels):
    """分析 before 和 after 向量的区分度"""
    n = len(before_vecs)
    norms_b = np.linalg.norm(before_vecs, axis=1, keepdims=True)
    norms_a = np.linalg.norm(after_vecs, axis=1, keepdims=True)
    norms_b = np.where(norms_b == 0, 1.0, norms_b)
    norms_a = np.where(norms_a == 0, 1.0, norms_a)

    b_norm = before_vecs / norms_b
    a_norm = after_vecs / norms_a

    # 同一 PR 的 before-after 余弦相似度
    same_pr_sims = (b_norm * a_norm).sum(axis=1)
    print(f"  同一 PR 的 before-after 余弦相似度:")
    print(f"    均值={same_pr_sims.mean():.4f}, 中位数={np.median(same_pr_sims):.4f}, "
          f"标准差={same_pr_sims.std():.4f}")

    # 按有无标签拆分
    if labels.sum() > 0 and (labels == 0).sum() > 0:
        merged_sims = same_pr_sims[labels == 1]
        unmerged_sims = same_pr_sims[labels == 0]
        print(f"    已合并 PR 的 before-after 相似度: 均值={merged_sims.mean():.4f}")
        print(f"    未合并 PR 的 before-after 相似度: 均值={unmerged_sims.mean():.4f}")

    # 欧氏距离
    euclidean_dists = np.linalg.norm(before_vecs - after_vecs, axis=1)
    print(f"  同一 PR 的 before-after 欧氏距离:")
    print(f"    均值={euclidean_dists.mean():.4f}, 中位数={np.median(euclidean_dists):.4f}, "
          f"标准差={euclidean_dists.std():.4f}")

    # 判断
    if same_pr_sims.mean() > 0.95:
        print(f"  ⚠️ Before 和 After 向量几乎一样 (相似度={same_pr_sims.mean():.4f})！")
        print(f"     可能原因: PR 修改幅度很小，或代码变化没有体现在 AST 路径上")
    elif same_pr_sims.mean() > 0.8:
        print(f"  ⚡ Before 和 After 向量相似度较高 ({same_pr_sims.mean():.4f})，但仍有区分")
    else:
        print(f"  ✅ Before 和 After 向量有明显差异 ({same_pr_sims.mean():.4f})")

    # 随机 PR 的 before-after 相似度（作为对照）
    indices = np.random.permutation(n)
    random_sims = (b_norm * a_norm[indices]).sum(axis=1)
    print(f"  随机配对的 before-after 余弦相似度 (对照):")
    print(f"    均值={random_sims.mean():.4f}, 中位数={np.median(random_sims):.4f}")

    if random_sims.mean() < same_pr_sims.mean():
        print(f"  ✅ 同 PR 的 before-after 比随机配对更相似，合理")
    else:
        print(f"  ⚠️ 同 PR 的 before-after 不比随机配对更相似，向量可能有问题")


def analyze_by_repo(before_vecs, after_vecs, pr_ids, pr_labels):
    """按仓库分组分析向量"""
    repo_vecs = defaultdict(list)
    for i, pr_id in enumerate(pr_ids):
        info = pr_labels.get(int(pr_id), {})
        repo = info.get('repo', 'unknown')
        concat = np.concatenate([before_vecs[i], after_vecs[i]])
        repo_vecs[repo].append(concat)

    for repo, vecs in repo_vecs.items():
        if len(vecs) < 3:
            continue
        vecs = np.array(vecs)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = vecs / norms
        sim_matrix = normalized @ normalized.T
        mask = ~np.eye(len(vecs), dtype=bool)
        all_sims = sim_matrix[mask]
        print(f"  {repo}: {len(vecs)} 个 PR, 内部余弦相似度: "
              f"均值={all_sims.mean():.4f}, 中位数={np.median(all_sims):.4f}")


# ========== 第 4 部分：向量范数分析 ==========

def analyze_vector_norms():
    """分析向量的 L2 范数分布"""
    print("\n" + "=" * 70)
    print("【第 3 部分】向量范数分析")
    print("=" * 70)

    vectors_path = config.TRAIN_VECTORS_PATH
    if not os.path.exists(vectors_path):
        print("❌ 向量文件不存在，跳过")
        return

    import torch
    data = torch.load(vectors_path, weights_only=False)
    before_vecs = data['before_vectors'].numpy()
    after_vecs = data['after_vectors'].numpy()

    for name, vecs in [("Before", before_vecs), ("After", after_vecs)]:
        norms = np.linalg.norm(vecs, axis=1)
        print(f"\n  {name} 向量 L2 范数:")
        print(f"    均值={norms.mean():.4f}, 中位数={np.median(norms):.4f}, "
              f"std={norms.std():.4f}")
        print(f"    最小={norms.min():.4f}, 最大={norms.max():.4f}")

        # 检查是否有零向量
        zero_count = (norms < 1e-6).sum()
        if zero_count > 0:
            print(f"    ⚠️ 有 {zero_count} 个零向量（或近乎零向量）！")
        else:
            print(f"    ✅ 无零向量")

        # 检查是否有退化
        if norms.std() < 1e-4:
            print(f"    ⚠️ 所有向量范数几乎相同，可能退化")
        else:
            print(f"    ✅ 范数分布有差异")


# ========== 第 5 部分：综合建议 ==========

def print_recommendations(before_arr, after_arr):
    """根据统计结果给出建议"""
    print("\n" + "=" * 70)
    print("【第 4 部分】综合建议")
    print("=" * 70)

    issues = []
    ok = []

    # 路径饱和度
    at_limit_before = (before_arr >= config.MAX_PATHS_PER_FILE).mean()
    at_limit_after = (after_arr >= config.MAX_PATHS_PER_FILE).mean()

    if at_limit_before > 0.5 or at_limit_after > 0.5:
        issues.append(
            f"⚠️ 超过 50% 的 PR 路径数达到上限 {config.MAX_PATHS_PER_FILE}，"
            f"大量有效路径被截断。建议将 MAX_PATHS_PER_FILE 调大到 500 或 1000"
        )
    elif at_limit_before > 0.3 or at_limit_after > 0.3:
        issues.append(
            f"⚡ 约 {max(at_limit_before, at_limit_after)*100:.1f}% 的 PR 路径数达到上限，"
            f"可考虑适当调大 MAX_PATHS_PER_FILE"
        )
    else:
        ok.append(
            f"✅ 路径饱和度合理，只有 {max(at_limit_before, at_limit_after)*100:.1f}% 达到上限"
        )

    # 路径数太少
    very_few_before = (before_arr < 20).mean()
    very_few_after = (after_arr < 20).mean()
    if very_few_before > 0.2 or very_few_after > 0.2:
        issues.append(
            f"⚠️ {max(very_few_before, very_few_after)*100:.1f}% 的 PR 路径数 < 20，"
            f"这些 PR 的向量可能信息不足"
        )

    # 平均路径数
    if before_arr.mean() < 50:
        issues.append(
            f"⚠️ Before 平均路径数仅 {before_arr.mean():.1f}，整体信息量可能不足，"
            f"可以考虑把 MAX_PATHS_PER_FILE 调小以控制质量，但这不是主要问题"
        )

    if issues:
        print("\n【发现的问题】")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    if ok:
        print("\n【正常项】")
        for item in ok:
            print(f"  {item}")

    if not issues:
        print("\n✅ 整体向量质量看起来合理！")

    print(f"\n当前配置: MAX_PATHS_PER_FILE = {config.MAX_PATHS_PER_FILE}")
    print(f"建议值范围: 200-500（根据你的饱和度数据决定）")


# ========== 主入口 ==========

def main():
    print("=" * 70)
    print("🔍 Code2Vec 向量质量诊断")
    print("=" * 70)
    print(f"路径上限: MAX_PATHS_PER_FILE = {config.MAX_PATHS_PER_FILE}")
    print(f"向量维度: CODE_VECTOR_DIM = {config.CODE_VECTOR_DIM}")
    print()

    # 第 1 部分：路径数统计（始终可运行）
    result = analyze_path_counts()
    if result is None:
        return
    before_arr, after_arr = result

    # 第 2 部分：向量多样性（需要向量文件）
    analyze_vector_diversity()

    # 第 3 部分：范数分析
    analyze_vector_norms()

    # 第 4 部分：综合建议
    print_recommendations(before_arr, after_arr)


if __name__ == '__main__':
    main()