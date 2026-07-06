步骤一：
根据lab1里面的raw数据筛选出来人类编写的pr保存到本地
步骤二：
根据上一步的信息，把每个pr的改之前的代码和改之后的代码爬下来保存到本地
然后把之前和之后的代码转化成cfg和ast保存到本地，要包含go，c++，c，java，js，python，tpyescript
步骤三：
根据上面的数据保存一下特征提取以下特征：
• 修改文件数量；
• 新增代码行数；
• 删除代码行数；
• Commit Message 长度；
• Pull Request 描述长度；
• AST 相关特征；
• CFG相关特征。
ast和cfg的特征主要统计每个类型节点的数量给序列化，采用聚合加均值的方法
步骤四
分别训练以下模型：
• Support Vector Machine（SVM）；
• Random Forest。
预测目标为：
• Pull Request 是否最终 Merge。
步骤五：模型评估
采用测试集评估模型性能，包括但不限于：
• Accuracy；
• Precision；
• Recall；
• F1-score；
• ROC-AUC。
进一步分析Random Forest 模型输出的特征重要性。
