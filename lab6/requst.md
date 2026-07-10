6 实验六 改进针对大模型生成代码的代码审查
6.1 实验基本信息
实验名称
实验学时
实验类型
实验性质
教学方式
改进针对大模型生成代码的代码审查
4 学时
综合性实验
必修
讲授+实践
对应课程目标 T1、T2、T3
6.2 实验目的
完成本实验后，学生应能够：
1. 理解AI 生成代码代码审查性能下降的主要原因。
2. 掌握不同代码上下文构建方法及其对代码审查任务的影响。
6 实验六 改进针对大模型生成代码的代码审查
30
3. 掌握Prompt Engineering 优化代码审查性能的方法。
4. 能够设计适用于AI生成代码的Prompt。
5. 能够分析不同 Prompt 和上下文组合对 Merge Prediction 和 Review Comment
Generation 任务的影响。
6. 能够提出针对AI生成代码的代码审查改进方案。
6.3 实验背景
实验五分析了传统机器学习模型、深度学习模型以及大语言模型在人类代码和AI
生成代码上的性能差异。实验结果表明，AI生成代码具有不同于人类代码的特征，使
得已有代码审查方法在Merge Prediction 和 Review Comment Generation 任务上的性
能出现下降。
与传统软件工程任务不同，AI生成代码通常更加依赖完整的软件工程上下文，仅
利用Pull Request 中的局部修改往往无法准确判断代码质量。同时，大语言模型具有较
强的上下文理解能力，其性能受到Prompt设计方式和上下文组织方式的显著影响。
因此，本实验将在实验五的基础上，通过设计不同粒度的软件工程上下文以及不同
类型的Prompt，对 AI 生成代码的代码审查方法进行改进，提高 Merge Prediction 和
Review Comment Generation 任务的性能。
6.4 实验原理
6.4.1 改进代码审查流程
针对AI生成代码的改进代码审查流程如下：
AI 生成代码
↓
上下文增强
↓
Prompt 优化
↓
大语言模型推理
↓
Merge Prediction
Review Comment Generation
相比实验五，本实验重点优化模型输入，而不是修改模型本身。
6 实验六 改进针对大模型生成代码的代码审查
31
6.4.2 上下文增强
为了帮助模型更全面地理解代码修改，本实验考虑引入更加丰富的软件工程上下
文，包括：
• Pull Request 描述；
• Commit Message；
• 修改前后的代码；
• 修改函数所在文件；
• 调用关系及相关函数；
• Issue 描述；
• 历史代码审查意见；
• Repository 级代码上下文。
不同上下文能够提供不同层次的信息，从而提高模型的推理能力。
6.4.3 Prompt 优化
Prompt 设计直接影响大语言模型的推理质量。本实验主要探索以下优化策略：
• Role-based Prompt；
• Few-shot Prompt；
• Chain-of-Thought Prompt；
• Self-Reflection Prompt；
• 多轮交互式Prompt。
通过不同Prompt 引导模型完成更加准确的代码审查。
6.4.4 代码审查任务
本实验继续关注以下两个任务：
1. Merge Prediction
结合增强后的上下文预测AI生成代码是否能够成功Merge。
2. Review Comment Generation
生成更加准确、更加具有指导意义的代码审查意见。
6 实验六 改进针对大模型生成代码的代码审查
32
6.5 实验环境
6.5.1 硬件环境
建议配置如下：
• CPU：8 Core 以上；
• 内存：16GB以上；
• GPU：推荐NVIDIA RTX3060 及以上；
• 网络环境：能够访问所使用的大语言模型服务。
6.5.2 软件环境
建议软件版本如下：
软件
推荐版本
Python
Transformers
3.10+
最新版本
OpenAI SDK 或兼容 SDK 最新版本
pandas
最新版本
numpy
matplotlib
最新版本
最新版本
6.6 实验内容
本实验需要完成以下任务：
1. 读取实验五中的AI生成代码数据集；
2. 构建不同粒度的软件工程上下文；
3. 设计多种Prompt 优化策略；
4. 调用大语言模型完成MergePrediction；
5. 调用大语言模型生成代码审查意见；
6. 比较不同上下文与Prompt 组合的性能，并分析改进效果。
6 实验六 改进针对大模型生成代码的代码审查
33
6.7 实验步骤
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
6 实验六 改进针对大模型生成代码的代码审查
34
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
6.9 实验思考
请结合实验过程回答以下问题：
1. 为什么Repository 级上下文能够提升代码审查性能？
2. 哪种Prompt 设计最适合 AI 生成代码的代码审查？为什么？
3. Prompt 优化与上下文增强分别解决了哪些问题？
4. 不同Prompt 组合是否适用于所有代码审查任务？
5. 针对AI 生成代码，未来还可以从哪些方面进一步提升代码审查效果？
7 实验七 实现VSCODE插件
35
6.10 实验拓展
完成基础实验后，可进一步尝试：
• 尝试Agent、多智能体等代码审查框架；
• 探索自动Prompt 优化与Prompt 搜索方法；
• 引入工具调用（Tool Use）、检索增强生成（RAG）等技术，提高AI生成代码代
码审查的准确率，为实验七开发VSCode插件提供更加稳定、高质量的代码审查
能力。