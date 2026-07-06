1 实验一 代码审查意见挖掘与需求理解
1.1 实验基本信息
实验名称
实验学时
实验类型
实验性质
教学方式
代码审查意见挖掘与需求理解
4 学时
综合性实验
必修
讲授+实践
对应课程目标 T1、T2
1.2 实验目的
完成本实验后，学生应能够：
1. 理解现代软件开发过程中PullRequest（PR）与 Code Review 的基本流程。
2. 熟悉GitHub 开源软件仓库中代码审查数据的组织形式。
3. 掌握Pull Request、Commit、Review、Comment 等对象之间的关系。
4. 能够利用GitHub API 或公开数据集获取代码审查数据。
5. 能够完成代码、文本及元数据等多源信息的抽取。
6. 掌握数据集统计分析方法，为后续模型训练提供数据基础。
1
1 实验一 代码审查意见挖掘与需求理解
2
1.3 实验背景
随着GitHub、GitLab 等协同开发平台的广泛应用，代码审查（Code Review）已
经成为现代软件开发过程中保证软件质量的重要环节。开发人员通常通过PullRequest
（PR）的形式提交代码修改，并由项目维护者或其他开发人员进行审查、讨论和修改建
议，最终决定是否将代码合并到主分支。
近年来，大语言模型（Large Language Models，LLMs）的快速发展进一步改变了
代码审查过程。一方面，大量代码已经由 GitHub Copilot、ChatGPT、Claude Code、
Cursor 等工具辅助生成；另一方面，越来越多的代码审查意见也开始由 AI 自动生成。
因此，现代代码审查已经逐渐形成了“人类开发者+AI开发者+AIReviewer”共同
参与的软件开发模式。
为了研究不同代码审查方法的性能，本课程首先需要构建一个包含丰富信息的数据
集，包括：
• Pull Request 基本信息；
• 提交代码；
• Code Review Comments；
• Review Decision；
• Merge 状态；
• 是否包含AI生成代码；
• 是否存在AIReviewer；
• 时间、标签等元数据。
该数据集将作为整个课程后续所有实验的数据基础。
1.4 实验原理
1.4.1 Pull Request 工作流程
现代开源软件通常采用PullRequest 的开发流程，如图所示。
1 实验一 代码审查意见挖掘与需求理解
3
此处插入Pull Request 工作流程图
Fork Repository
↓
Commit Changes
↓
Open Pull Request
↓
Code Review
↓
Discussion
↓
Merge / Close
Pull Request 包含的不仅仅是代码修改，还包括：
• Commit 信息；
• 修改文件；
• Issue 引用；
• Code Review；
• Discussion；
• Merge Decision。
1.4.2 代码审查流程
代码审查主要包含以下几个阶段：
1. 提交代码修改；
2. Reviewer 阅读代码；
3. Reviewer 提交修改建议；
4. Developer 回复意见；
5. 修改代码；
6. 最终决定是否Merge。
代码审查不仅能够发现软件缺陷，还能够提高代码可维护性、统一编码规范，并促
进团队成员之间的知识共享。
1 实验一 代码审查意见挖掘与需求理解
4
1.4.3 Pull Request 数据结构
一次Pull Request 通常包含多个对象，其关系如图所示。
Pull Request
￿￿￿ Commits
￿￿￿ Changed Files
￿￿￿ Reviews
￿￿￿ Review Comments
￿￿￿ Issue Comments
￿￿￿ Labels
这些对象共同构成了代码审查任务的重要数据来源。
1.4.4 代码审查任务
本课程主要关注两个典型任务：
1. Merge Prediction
判断一个Pull Request 是否最终能够被项目接受。
2. Review Comment Generation
根据代码修改内容自动生成合理的代码审查意见。
后续实验均围绕这两个任务展开。
1.5 实验环境
1.5.1 硬件环境
建议配置如下：
• CPU：4 Core 以上；
• 内存：8GB以上；
• 推荐网络连接GitHub。
1 实验一 代码审查意见挖掘与需求理解
5
1.5.2 软件环境
建议软件版本如下：
1.6 实验内容
软件
推荐版本
Python
Git
VSCode
3.10+
Latest
Latest
Jupyter Notebook Latest
pandas
最新版本
requests
PyGithub
matplotlib
最新版本
最新版本
最新版本
本实验需要完成以下任务：
1. 熟悉GitHub Pull Request 的组织形式；
2. 获取指定开源项目的PullRequest 数据；
3. 获取Pull Request 对应 Review；
4. 获取Code Review Comments；
5. 获取Commit 信息；
6. 获取Merge 状态；
7. 提取Pull Request 标签；
8. 判断是否存在AIReviewer；
9. 判断是否可能由AI生成代码；
10. 对整个数据集进行统计分析。
1.7 实验步骤
1.7.1 步骤一：获取 Pull Request
选择5个大型开源软件，需要有AI生成的代码，有AI做代码审查的PR。
利用GitHub API，每个项目获取 300 条 PR 数据：
1 实验一 代码审查意见挖掘与需求理解
6
• PRID
• Title
• Body
• Author
• Created Time
• Merge Status
1.7.2 步骤二：获取 Code Review
继续获取：
• Reviewer
• Review Decision
• Review Time
• Review Comment
1.7.3 步骤三：获取代码修改
提取：
• 修改文件
• 修改函数
• Diff
• Commit Message
1.7.4 步骤四：提取特征
完成如下信息统计：
• PR长度
• 修改文件数
• Reviewer 数量
• Comment 数量
1 实验一 代码审查意见挖掘与需求理解
7
• 是否Merge
• Label 数量
• 是否包含AIReviewer
• 是否存在AIGenerated Code
• PR 使用的编程语言（主语言、语言种类、语言列表）
1.7.5 步骤五：数据分析
完成统计分析，包括但不限于：
• Merge 与 Non-Merge 数量统计；
• Review Comment 数量分布；
• Label 分布；
• Reviewer 数量分布；
• PR长度分布；
• AI 与Human PR 数量比较。
建议绘制柱状图、饼图、直方图等可视化结果。
1.8 实验结果
实验报告中，学生应展示以下实验结果：
1. 成功获取的Pull Request 数据样例；
2. 成功获取的Review Comments 样例；
3. 成功完成数据预处理的阳历；
4. 数据统计结果；
5. 可视化分析结果。
2 实验一 代码审查意见挖掘与需求理解
8
1.9 实验思考
请结合实验过程回答以下问题：
1. Pull Request 与 Commit 有何区别？
2. 为什么Code Review 能够提高软件质量？
3. Merge Prediction 可以应用于哪些实际场景？
4. AI Reviewer 与 Human Reviewer 的审查方式可能有哪些差异？
5. 数据集是否存在类别不平衡问题？如何解决？
1.10 实验拓展
完成基础实验后，可进一步尝试：
• 分析不同开源社区（如Kubernetes、PyTorch、VSCode）之间代码审查流程的差
异；
• 构建适用于整个课程的数据仓库，为后续机器学习和深度学习实验提供统一的数
据输入。