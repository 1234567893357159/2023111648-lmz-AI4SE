步骤一：
把lab2里面的pr分成并划分训练集、验证集和测试集，保存到本地，只用保存成三个json文件就行，每个json文件包含pr_id、before_code的路径文件夹、after_code路径文件夹（文件夹只想lab2里面的）、label（指的是是否合并）
步骤二：
使用Code2Vec技术把before_code和after_code转化成向量，保存到本地，tree-sitter的使用可以参考lab2里面的test_language_parsing
步骤三：
将上面获得的向量处理，主要是针对一个pr有多个代码更改哦的情况，分别将前后向量聚，包括平均值和求和，生成新的一个pr值对应一个向量（这个向量是前后聚合完的向量的拼接）
然后分别使用随机森林，svm和多层感知机MLP训练，然后用测试集评估模型性能，包括但不限于：
• Accuracy；
• Precision；
• Recall；
• F1-score；
• ROC-AUC。
步骤4:
利用预训练 CodeBERT 模型完成模型微调，分别针对 Merge Prediction 任务和
Review Comment Generation 任务训练模型