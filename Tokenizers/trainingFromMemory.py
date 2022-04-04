# @Time : 2022-04-04 14:12
# @Author : Phalange
# @File : trainingFromMemory.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
import datasets
import requests
import gzip

print("-----------------------------------------")
print("注意，这个脚本是不能运行的！")
print("------------------------------------------")

tokenizer = Tokenizer(models.Unigram())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()


trainer = trainers.UnigramTrainer(
    vocab_size=20000,
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<PAD>","<BOS>","<EOS>"],
)

"""
第一种常见的方式，使用list,tuple,np.array来迭代

"""
# First few lines of the "Zen of Python" https://www.python.org/dev/peps/pep-0020/
data = [
    "Beautiful is better than ugly."
    "Explicit is better than implicit."
    "Simple is better than complex."
    "Complex is better than complicated."
    "Flat is better than nested."
    "Sparse is better than dense."
    "Readability counts."
]
# data的格式只要可迭代就行，可以是list，tuple，np.array
tokenizer.train_from_iterator(data, trainer=trainer)


"""
使用Datasets库
"""
# 1.加载数据集  不过现在它会报错误。。就是说连接不上网。。先不研究了
dataset = datasets.load_dataset(    "wikitext", "wikitext-103-raw-v1", split="train+test+validation")

# 2.构建迭代器,为了速度快，可以使用batch_size的方法去读取数据集
def batch_iterator(batch_size=1000):
    for i in range(0,len(dataset),batch_size):
        yield dataset[i:i+batch_size]["text"]

#tokenizer.train_from_iterator(batch_iterator(),trainer=trainer,length=len(dataset))


"""
Using gzip files
"""

with gzip.open("data/my-file.0.gz", "rt") as f:
    tokenizer.train_from_iterator(f, trainer=trainer)

# 如果想训练多个

files = ["data/my-file.0.gz", "data/my-file.1.gz", "data/my-file.2.gz"]

def gzip_iterator():
    for path in files:
        with gzip.open(path, "rt") as f:
            for line in f:
                yield line

tokenizer.train_from_iterator(gzip_iterator(), trainer=trainer)