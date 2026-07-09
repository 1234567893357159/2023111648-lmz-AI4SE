阶段一：数据准备（复用实验三成果）
要做什么：读取实验三构建好的测试数据集。数据集里必须包含：diff（代码变更）、commit_message、pr_description（PR描述），以及真实标签（是否真的被合并了，用于算指标）。

关键动作：写一个Python脚本（pandas）把数据加载进来，按ID索引，准备循环喂给大模型。

阶段二：构建代码上下文（核心变量1）
要做什么：针对同一条数据，构造4种不同长度的文本输入，用来测试“信息量”对结果的影响。指导书要求至少做以下组合：

仅Diff（最基础）。
Diff + PR描述。
Diff + Commit Message。
Diff + 其他信息（建议你把文件名、改动的函数名加上，或者加上历史评论）。
技术实现：写一个函数 build_context(data, context_type)，返回拼接好的长字符串。

阶段三：设计Prompt（核心变量2）
要做什么：针对每一种上下文，你都要设计4种指令模板（Prompt），分别调用模型。总共就是 4（上下文）× 4（Prompt）= 16组实验。

四种Prompt的具体写法建议：

Zero-shot：直接问“请判断以下代码变更能否被合并？回答是或否。”和“请写出审查意见。”

Few-shot：在指令前加上2个“好的审查样例”（比如：Diff A + 正确意见，Diff B + 正确意见），让模型模仿。

Chain-of-Thought (CoT)：加上“请一步一步分析：1.代码逻辑是否正确？2.是否有安全隐患？3.格式是否规范？最后得出结论。”

Role-based：加上“你现在是Apache开源项目的资深Committer，拥有严格的代码质量标准，请审查...”

阶段四：模型推理（调用API）
要做什么：调用大语言模型（根据你的环境，可能是OpenAI SDK，或者国内DeepSeek/Qwen的API）。针对每条数据，你需要让模型完成两个动作：

Merge Prediction：强制模型输出结构化结果（建议用JSON格式，如 {"decision": "Yes/No", "confidence": 0.9}），方便你提取。
Review Comment Generation：让模型生成一段自然语言的审查意见文本。
记录：务必记录推理时间（Latency），实验报告要用的。

阶段五：结果量化评估（最费脑力的部分）
对于 Merge Prediction（分类任务）：把模型输出的 Yes/No 和数据集里的真实标签对比，算出 Accuracy、Precision、Recall、F1-score。你可以写个sklearn的分类报告。

对于 Review Comment Generation（生成任务）：把模型生成的评论和实验三数据集里人类原本写的评论做对比，计算 BLEU 和 ROUGE 分数（用 nltk 或 evaluate 库）。

对比分析：做两张热力图或柱状图，一张横轴是不同Prompt，一张横轴是不同上下文，看哪个组合分数最高。

阶段六：与实验三对比（升华点）
要做什么：拿出你实验三训练好的CodeBERT或深度学习模型的结果（F1值、BLEU值），和你本实验表现最好的那个Prompt组合做横向对比。

结论预设：大模型可能在“生成评论的流畅度”上赢，但在“精确预测合并”上可能不如微调的小模型（或者反过来），如实分析即可。