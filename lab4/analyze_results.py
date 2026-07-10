"""
分析 lab4 merge_prediction 结果的解析成功率 + 分类指标
直接运行即可: python analyze_results.py
"""
import json
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SELECTED_PRS_PATH = os.path.join(BASE_DIR, "data", "selected_prs.json")

FILES = [
    "merge_prediction_zero_shot.json",
    "merge_prediction_few_shot.json",
    "merge_prediction_cot.json",
    "merge_prediction_role_based.json",
]

CONTEXT_TYPES = ["diff_only", "diff_pr_desc", "diff_commit", "diff_extra"]


def load_ground_truth():
    """从 selected_prs.json 加载真实标签，返回 {pr_id: label} 映射"""
    if not os.path.exists(SELECTED_PRS_PATH):
        print(f"  [警告] 找不到 {SELECTED_PRS_PATH}，无法计算分类指标")
        return {}
    with open(SELECTED_PRS_PATH, "r", encoding="utf-8") as f:
        prs = json.load(f)
    return {pr["pr_id"]: pr["label"] for pr in prs}


def compute_classification_metrics(data, label_map):
    """根据真实标签计算分类指标，返回 (y_true, y_pred, y_score, metrics_dict)"""
    y_true, y_pred, y_score = [], [], []
    for r in data:
        pr_id = r.get("pr_id")
        if pr_id not in label_map:
            continue
        y_true.append(label_map[pr_id])

        decision = r.get("output_decision", "Unknown")
        if decision == "Yes":
            y_pred.append(1)
            y_score.append(1.0)
        elif decision == "No":
            y_pred.append(0)
            y_score.append(0.0)
        else:
            y_pred.append(0)
            y_score.append(0.5)

    n = len(y_true)
    if n == 0:
        return None

    if len(set(y_true)) < 2:
        return {
            "n_samples": n,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "note": "只有单一类别",
        }

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = None

    return {
        "n_samples": n,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
    }


def analyze_one_file(filepath, label_map):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    parse_ok = sum(1 for r in data if r.get("parse_error") == False)
    parse_fail = sum(1 for r in data if r.get("parse_error") == True)
    unknown = sum(1 for r in data if r.get("output_decision") == "Unknown")
    yes = sum(1 for r in data if r.get("output_decision") == "Yes")
    no = sum(1 for r in data if r.get("output_decision") == "No")

    fail_records = [r for r in data if r.get("parse_error") == True]
    fail_yes = sum(1 for r in fail_records if r.get("output_decision") == "Yes")
    fail_no = sum(1 for r in fail_records if r.get("output_decision") == "No")
    fail_unknown = sum(1 for r in fail_records if r.get("output_decision") == "Unknown")

    ctx_stats = {}
    for ct in CONTEXT_TYPES:
        ctx_data = [r for r in data if r.get("context_type") == ct]
        ctx_total = len(ctx_data)
        ctx_ok = sum(1 for r in ctx_data if r.get("parse_error") == False)
        ctx_unk = sum(1 for r in ctx_data if r.get("output_decision") == "Unknown")
        ctx_cls = compute_classification_metrics(ctx_data, label_map)
        ctx_stats[ct] = {
            "total": ctx_total,
            "ok": ctx_ok,
            "fail": ctx_total - ctx_ok,
            "unknown": ctx_unk,
            "cls_metrics": ctx_cls,
        }

    avg_latency = sum(r.get("latency", 0) for r in data) / total if total > 0 else 0

    overall_cls = compute_classification_metrics(data, label_map)

    return {
        "total": total,
        "parse_ok": parse_ok,
        "parse_fail": parse_fail,
        "unknown": unknown,
        "yes": yes,
        "no": no,
        "fail_yes": fail_yes,
        "fail_no": fail_no,
        "fail_unknown": fail_unknown,
        "ctx_stats": ctx_stats,
        "avg_latency": avg_latency,
        "cls_metrics": overall_cls,
    }


def _format_cls_metric(val):
    if val is None:
        return "N/A"
    return f"{val:.4f}"


def main():
    print("=" * 75)
    print("  Lab4 Merge Prediction 结果解析状态 + 分类指标分析")
    print("=" * 75)

    label_map = load_ground_truth()
    if label_map:
        print(f"  已加载 {len(label_map)} 条真实标签 (selected_prs.json)")
    else:
        print(f"  未找到真实标签，将跳过分类指标计算")

    all_stats = {}

    for fname in FILES:
        filepath = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(filepath):
            print(f"\n  [跳过] 文件不存在: {fname}")
            continue

        prompt_type = fname.replace("merge_prediction_", "").replace(".json", "")
        stats = analyze_one_file(filepath, label_map)
        all_stats[prompt_type] = stats

        print(f"\n{'─' * 75}")
        print(f"  Prompt 类型: {prompt_type}")
        print(f"{'─' * 75}")

        print(f"\n  📊 整体解析状况:")
        print(f"     总记录数:        {stats['total']}")
        print(f"     解析成功 (JSON):  {stats['parse_ok']:>4}  ({stats['parse_ok']/stats['total']*100:5.1f}%)")
        print(f"     解析失败:         {stats['parse_fail']:>4}  ({stats['parse_fail']/stats['total']*100:5.1f}%)")
        print(f"       └ 降级匹配到 Yes: {stats['fail_yes']:>4}")
        print(f"       └ 降级匹配到 No:  {stats['fail_no']:>4}")
        print(f"       └ 完全无法解析:   {stats['fail_unknown']:>4}")

        print(f"\n  📈 预测分布:")
        print(f"     Yes:     {stats['yes']:>4}  ({stats['yes']/stats['total']*100:5.1f}%)")
        print(f"     No:      {stats['no']:>4}  ({stats['no']/stats['total']*100:5.1f}%)")
        print(f"     Unknown: {stats['unknown']:>4}  ({stats['unknown']/stats['total']*100:5.1f}%)")

        print(f"\n  🔍 按上下文类型拆分 (解析状况):")
        print(f"     {'上下文':<18} {'总数':<6} {'解析成功':<10} {'成功率':<10} {'Unknown':<8}")
        print(f"     {'─'*52}")
        for ct in CONTEXT_TYPES:
            cs = stats["ctx_stats"][ct]
            print(f"     {ct:<18} {cs['total']:<6} {cs['ok']:<10} {cs['ok']/cs['total']*100:<9.1f}% {cs['unknown']:<8}")

        print(f"\n  ⏱ 平均延迟: {stats['avg_latency']:.1f}s")

        cls = stats.get("cls_metrics")
        if cls:
            print(f"\n  🎯 整体分类指标 (vs 真实标签):")
            print(f"     样本数:    {cls['n_samples']}")
            print(f"     Accuracy:  {_format_cls_metric(cls['accuracy'])}")
            print(f"     Precision: {_format_cls_metric(cls['precision'])}")
            print(f"     Recall:    {_format_cls_metric(cls['recall'])}")
            print(f"     F1-score:  {_format_cls_metric(cls['f1'])}")
            print(f"     ROC-AUC:   {_format_cls_metric(cls['roc_auc'])}")
            if cls.get("note"):
                print(f"     ⚠ {cls['note']}")

            print(f"\n  🔍 按上下文类型拆分 (分类指标):")
            print(f"     {'上下文':<18} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8} {'N':<6}")
            print(f"     {'─'*62}")
            for ct in CONTEXT_TYPES:
                cs = stats["ctx_stats"][ct]
                cm = cs.get("cls_metrics")
                if cm is None:
                    print(f"     {ct:<18} {'N/A':>8}")
                elif cm.get("note"):
                    print(f"     {ct:<18} {cm['accuracy']:<8.4f} ({cm['note']})")
                else:
                    print(f"     {ct:<18} {_format_cls_metric(cm['accuracy']):<8} "
                          f"{_format_cls_metric(cm['precision']):<8} "
                          f"{_format_cls_metric(cm['recall']):<8} "
                          f"{_format_cls_metric(cm['f1']):<8} "
                          f"{_format_cls_metric(cm['roc_auc']):<8} {cm['n_samples']:<6}")

    print(f"\n\n{'=' * 75}")
    print(f"  跨 Prompt 类型汇总对比")
    print(f"{'=' * 75}")
    print(f"\n  {'Prompt':<15} {'总数':<6} {'解析成功':<10} {'成功率':<10} {'Unknown':<10} {'预测Yes':<8} {'预测No':<8} {'平均延迟':<10}")
    print(f"  {'─' * 80}")
    for pt in ["zero_shot", "few_shot", "cot", "role_based"]:
        if pt not in all_stats:
            continue
        s = all_stats[pt]
        print(f"  {pt:<15} {s['total']:<6} {s['parse_ok']:<10} {s['parse_ok']/s['total']*100:<9.1f}% {s['unknown']:<10} {s['yes']:<8} {s['no']:<8} {s['avg_latency']:<9.1f}s")

    print(f"\n\n{'=' * 75}")
    print(f"  Unknown 记录的来源分析（按上下文 × Prompt）")
    print(f"{'=' * 75}")
    header_col = "Prompt \\ 上下文"
    print(f"\n  {header_col:<20}", end="")
    for ct in CONTEXT_TYPES:
        print(f"{ct:<16}", end="")
    print(f"  {'合计':<6}")
    print(f"  {'─' * 85}")
    for pt in ["zero_shot", "few_shot", "cot", "role_based"]:
        if pt not in all_stats:
            continue
        print(f"  {pt:<20}", end="")
        total_unk = 0
        for ct in CONTEXT_TYPES:
            cs = all_stats[pt]["ctx_stats"][ct]
            print(f"{cs['unknown']:<16}", end="")
            total_unk += cs["unknown"]
        print(f"  {total_unk:<6}")

    if label_map:
        print(f"\n\n{'=' * 75}")
        print(f"  跨 Prompt × 上下文 分类指标对比")
        print(f"{'=' * 75}")
        for metric_name, metric_key in [("Accuracy", "accuracy"), ("Precision", "precision"),
                                         ("Recall", "recall"), ("F1-score", "f1"),
                                         ("ROC-AUC", "roc_auc")]:
            print(f"\n  📊 {metric_name}:")
            print(f"  {'Prompt':<15}", end="")
            for ct in CONTEXT_TYPES:
                print(f"{ct:<16}", end="")
            print()
            print(f"  {'─' * 80}")
            for pt in ["zero_shot", "few_shot", "cot", "role_based"]:
                if pt not in all_stats:
                    continue
                print(f"  {pt:<15}", end="")
                for ct in CONTEXT_TYPES:
                    cs = all_stats[pt]["ctx_stats"][ct]
                    cm = cs.get("cls_metrics")
                    if cm is None or cm.get(metric_key) is None:
                        print(f"{'N/A':<16}", end="")
                    else:
                        print(f"{cm[metric_key]:<16.4f}", end="")
                print()

    print(f"\n✅ 分析完成!")


if __name__ == "__main__":
    main()