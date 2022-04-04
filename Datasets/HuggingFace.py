# @Time : 2022-04-04 21:05
# @Author : Phalange
# @File : HuggingFace.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

from datasets import load_dataset_builder
from datasets import load_dataset
from datasets import get_dataset_config_names
from datasets import load_dataset
from datasets import get_dataset_split_names

# load_dataset_builder可以不下载数据就能查看数据的属性
dataset_builder = load_dataset_builder('imdb')
print(dataset_builder.cache_dir)
print(dataset_builder.info.features)
print(dataset_builder.info.splits)


dataset = load_dataset("glue",'mrpc',split='train',cache_dir="../cache/")
# Use get_dataset_config_names to retrieve a list of all the possible configurations available to your dataset
configs = get_dataset_config_names("glue")
print(configs)

dataset = load_dataset('glue', 'sst2',cache_dir="../cache/")
print(dataset)

# 和get_dataset_config_names类似：You can list the split names for a dataset, or a specific configuration, with the datasets.get_dataset_split_names() method
print(get_dataset_split_names('sent_comp'))

print(get_dataset_split_names('sent_comp'))
