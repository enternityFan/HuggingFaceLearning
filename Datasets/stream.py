# @Time : 2022-04-05 11:13
# @Author : Phalange
# @File : stream.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

"""
stream听起来有点酷的，不用等全部数据下载下来，就开始迭代，边用边下
当
    1.数据太大了，下载很慢
    2.硬盘内存不够
这两种情况时，非常好用

"""
from datasets import load_dataset,interleave_datasets

# stream模式下创造的实例是Datasets.IterableDataset
dataset = load_dataset('oscar','unshuffled_deduplicated_en',split='train',streaming=True)
print(next(iter(dataset)))


"""
和常规的datasets.Dataset类相似，可以shuffle这个IterableDataset
"""
shuffled_dataset = dataset.shuffle(seed=42,buffer_size=10_1000) # 这个buffer-size的数据设置的。。。有点抽象啊这个下划线

# 有时，想去reshuffle在每次迭代之后，这会让你设置不同的种子为了每次迭代。
for epoch in range(10):
    shuffled_dataset.set_epoch(epoch)
    for example in shuffled_dataset:
        pass

"""
Split dataset

take 和 skip 防止将来调用 shuffle，因为它们按分片的顺序锁定。您应该在拆分数据集之前对其进行洗牌。

"""
# 返回n个例子
dataset_head = dataset.take(2)
list(dataset_head)
# 删除第n个例子，并且返回剩余的
train_dataset = shuffled_dataset.skip(1000)

"""
Interleave

datasets.interleave_datasets可以结合也给datasets.InterableDataset和其他的数据集一起

"""
en_dataset = load_dataset('oscar', "unshuffled_deduplicated_en", split='train', streaming=True)
fr_dataset = load_dataset('oscar', "unshuffled_deduplicated_fr", split='train', streaming=True)
multilingual_dataset = interleave_datasets([en_dataset,fr_dataset])
print(list(multilingual_dataset.take(2)))

# 可以设置采样率
multilingual_dataset_with_oversampling = interleave_datasets([en_dataset, fr_dataset], probabilities=[0.8, 0.2], seed=42)
list(multilingual_dataset_with_oversampling.take(2))


"""
Rename remove cast就不说了吧。。。有在process里面
maping也不说了process都有的。。
"""