# tensorflow_models_nets

通过机器学习来完成音乐评分功能

1、从原始音频中提取特征
makeOnsetsBatch.py
2、将样本分成训练集和验证集
moveFiles.py
3、将样本文件名称和类别对齐
create_labels_files.py
4、将样本数据生成TfRecord文件
create_tf_record.py