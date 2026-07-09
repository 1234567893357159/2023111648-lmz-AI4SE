步骤一：数据筛选

从 lab1 的 1500 个 PR 数据集中，筛选出AI 生成代码对应的 PR
在 lab1 中已经通过关键词（Copilot、ChatGPT、Claude、Cursor 等）和启发式规则标注了 has_ai_generated_code 字段，这里就是筛选那些标记为 AI 生成的 PR
步骤二 ~ 四：用已有模型测试 AI 代码

把 lab2 训练好的 SVM / Random Forest 模型拿来，对 AI 代码做 Merge Prediction
把 lab3 训练好的 Code2Vec / CodeBERT 模型拿来，做 Merge Prediction + Review Comment Generation
把 lab4 调好的 大语言模型（Ollama qwen2.5-coder:7b） 拿来，用同样的 Prompt 对 AI 代码做推理
步骤五：性能分析

对比每个模型在 人类代码 vs AI 代码 上的指标变化
分析性能下降的原因
总结 AI 生成代码给代码审查带来的新挑战