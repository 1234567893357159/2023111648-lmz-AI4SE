"""
分析 lab4 merge_prediction 结果的解析成功率
直接运行即可: python analyze_results.py
"""
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results")

FILES = [
    "merge_prediction_zero_shot.json",
    "merge_prediction_few_shot.json",
    "merge_prediction_cot.json",
    "merge_prediction_role_based.json",
]

CONTEXT_TYPES = ["diff_only", "diff_pr_desc", "diff_commit", "diff_extra"]


def analyze_one_file(filepath):
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
        ctx_stats[ct] = {
            "total": ctx_total,
            "ok": ctx_ok,
            "fail": ctx_total - ctx_ok,
            "unknown": ctx_unk,
        }

    avg_latency = sum(r.get("latency", 0) for r in data) / total if total > 0 else 0

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
    }


def main():
    print("=" * 75)
    print("  Lab4 Merge Prediction 结果解析状态分析")
    print("=" * 75)

    all_stats = {}

    for fname in FILES:
        filepath = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(filepath):
            print(f"\n  [跳过] 文件不存在: {fname}")
            continue

        prompt_type = fname.replace("merge_prediction_", "").replace(".json", "")
        stats = analyze_one_file(filepath)
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

        print(f"\n  🔍 按上下文类型拆分:")
        print(f"     {'上下文':<18} {'总数':<6} {'解析成功':<10} {'成功率':<10} {'Unknown':<8}")
        print(f"     {'─'*52}")
        for ct in CONTEXT_TYPES:
            cs = stats["ctx_stats"][ct]
            print(f"     {ct:<18} {cs['total']:<6} {cs['ok']:<10} {cs['ok']/cs['total']*100:<9.1f}% {cs['unknown']:<8}")

        print(f"\n  ⏱ 平均延迟: {stats['avg_latency']:.1f}s")

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

    print(f"\n✅ 分析完成!")


if __name__ == "__main__":
    main()