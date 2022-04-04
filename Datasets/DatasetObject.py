# @Time : 2022-04-04 21:20
# @Author : Phalange
# @File : DatasetObject.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D
from datasets import load_dataset
dataset = load_dataset('glue', 'mrpc', split='train',cache_dir="../cache/")

# data.info返回简短的数据集的介绍
print(dataset.info)
print(dataset.split)
print(dataset.description)
print(dataset.citation)
print(dataset.homepage)
print(dataset[0])
print(dataset.shape)
print(dataset.num_rows)
print(len(dataset)) # dataset.shape[0]
print(dataset.column_names)
print(dataset.features)

# 关于标签
print(dataset.features['label'].num_classes)
print(dataset.features['label'].names)
print(dataset[:3])
print(dataset[[1,3,5]])

print(dataset['sentence1'][:3])
"""
A single row like dataset[0] returns a Python dictionary of values.
A batch like dataset[5:10] returns a Python dictionary of lists of values.
A column like dataset['sentence1'] returns a Python list of values.
"""