6.7.1 步骤一：数据准备
读取实验五中的AI生成代码数据，并准备代码审查任务所需的输入。
6.7.2 步骤二：上下文构建
分别构建不同粒度的软件工程上下文，包括：
• Diff；
• Diff + Pull Request 描述；
• Diff + Repository 上下文；
• Diff + Issue 信息；
• 完整软件工程上下文。
6.7.3 步骤三：Prompt 设计
针对不同上下文设计以下Prompt：
• Role-based Prompt；
• Few-shot Prompt；
• Chain-of-Thought Prompt；
• Self-Reflection Prompt。
6.7.4 步骤四：模型推理
利用设计好的上下文和Prompt调用大语言模型，分别完成：
• Merge Prediction；
• Review Comment Generation。
记录模型输出结果及推理时间。
6.7.5 步骤五：结果分析
统计不同方法的实验结果，包括但不限于：
• Accuracy；
• Precision；
• Recall；
• F1-score；
• BLEU；
• ROUGE；
• 推理时间。
分析不同上下文和Prompt设计对代码审查性能的影响，并与实验五进行对比。
6.8 实验结果
实验报告中，学生应展示以下实验结果：
1. 不同上下文构建样例；
2. 不同Prompt 设计样例；
3. Merge Prediction 实验结果；
4. 自动生成代码审查意见样例；
5. 不同上下文性能比较；
6. 不同Prompt 设计性能比较；
7. 与实验五实验结果的对比分析。