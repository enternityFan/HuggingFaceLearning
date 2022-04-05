# @Time : 2022-04-05 9:21
# @Author : Phalange
# @File : Process.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

"""
Process教程主要要学以下几点：
    1。重新排序行并且分割数据集
    2。重命名和移除列，并且其他常见的列操作
    3.应用处理函数对于数据集中的每一个例子
    4.合并数据集
    5.应用一个常见的格式转换
    6.保存和导出处理后的数据集

全部的处理方法都返回了一个新的datasets.Dataset.修改不是就地完成的。小心覆盖以前的数据集！
"""

from datasets import load_dataset,Dataset
import datasets
from datasets import ClassLabel,Value,Audio,concatenate_datasets,load_from_disk
from transformers import BertTokenizerFast,BertTokenizer
from random import randint
from transformers import pipeline


dataset = load_dataset('glue','mrpc',split='train',cache_dir='../cahche/')

# 1.sort
print(dataset['label'][:10])
sorted_dataset = dataset.sort('label')
print(sorted_dataset['label'][:10])
print(sorted_dataset['label'][-10:])

# 2.Shuffle置随机数
shuffled_dataset = sorted_dataset.shuffle(seed=42)
print(shuffled_dataset['label'][:10])

# 3.选择和过滤
# 3.1通过select，返回一个根据索引列表的行
small_dataset = dataset.select([0,10,20,30,40,50])
print(len(small_dataset))
# 3.2通过filter,返回匹配特殊条件的行
start_with_ar = dataset.filter(lambda example: example['sentence1'].startswith('Ar'))
print(len(start_with_ar))
print(start_with_ar['sentence1']) # 可以看出结果的行的第一个单词都是以Ar开头的

# filter也可以通过索引来过滤
even_dataset = dataset.filter(lambda example,idx: idx % 2 ==0,with_indices=True)
print(len(even_dataset))
print(len(dataset) / 2)

"""
split
这个分割自动是开启shuffled功能的，可以通过传入参数shuffle=False来阻止shuffled
"""
dataset.train_test_split(test_size=0.1)
print(0.1 * len(dataset))

"""
shard
分片操作把一个非常大的dataset分割为预定义数目的块。
 num_shards变量决定shards的数目，就是要分成几块
 也需要提供想返回的索引值
"""
dataset = load_dataset('imdb',split='train')
print(dataset)
dataset.shard(num_shards=4,index=0)
print(25000 / 4)

# rename操作,与原始列相关的特征，实际上被移动到新列名下，而不是仅仅在原地替换原始列
print(dataset)
dataset = dataset.rename_column("sentence1","sentenceA")
dataset = dataset.rename_column("sentence2","sentenceB")
print(dataset)

# remove 如果需要移除一行，直接提供名字，如果需要移除很多行，就需要提供列表
dataset = dataset.rename_columns("label")
print(dataset)
dataset = dataset.rename_columns(['sentence1','sentence2'])
print(dataset)


"""
cast 可以改变一列或者更多列的特征，会产生新的数据集。
下面例子示例了怎么改变datasets.ClassLabel 和 datasets.Value
注意，当使用cast的时候，需要保证原始的特征和新的特征是匹配的。例如如果原来的列只包含0或者1则Value('int32')和Value('bool')是匹配的
"""
print(dataset.features)
new_features = dataset.features.copy()
new_features["label"] = ClassLabel(names=['negative','positive'])
new_features["idx"] = Value('int64')
dataset = dataset.cast(new_features)
print(dataset.features)
print(dataset.features)
dataset = dataset.cast_column("audio",Audio(sampling_rate=16000))
print(dataset.features)

"""
Flatten
有时一列是一个嵌套结构包含几种数据类型，flatten()可以提取这些子模块到分离出来的他们自己的列
"""
dataset = load_dataset('squad',split='train')
print(dataset.features)
# answers有两个子的，text和answer_start，可以给他展开
flat_dataset = dataset.flatten()
print(flat_dataset)

"""
Map  这是最有用的应用

最初提出这个是为了加速函数计算吧，可以应用一个处理函数到每个example in a dataset,处理函数甚至可以创造新的行或者列

"""

def add_prefix(example):
    """
    给dataset中的没有给example的sentence1添加一个句首"My sentence:"
    :param example:
    :return: exampleJ
    """
    example['sentence1'] = 'My sentence:' + example['sentence1']
    return example

updated_dataset = small_dataset.map(add_prefix)
updated_dataset['sentence1'][:5]

# 使用map移除列，只移除向map提供的列，并且在移除这些列之前，map函数是可以使用这些列中的值的
#remove_columns的速度比较快，因为他不复制剩余行的数据
updated_dataset = dataset.map(lambda example:{'new_sentence':example['sentence1']},remove_columns=['sentence1'])
print(updated_dataset.column_names)

# 也可以通过设置with_indices=True来与索引一起使用
# 也可以通过使用with_rank=True来排序进程？
updated_dataset = dataset.map(lambda  example,idx:{'sentence2':f'{idx}:' + example['sentence2']},with_indices=True)
print(updated_dataset['sentence2'][:5])


"""
多进程处理和多GPU处理就给略了先。。。
"""
# map还可以进行批处理，同故宫设置 batched = True
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
encoded_dataset = dataset.map(lambda examples:tokenizer(examples['sentence1']),batched=True)
print(encoded_dataset.column_names)
print(encoded_dataset[0])

"""
分割long examples
"""
def chunk_examples(examples):
    chunks = []
    for sentence in examples['sentece1']:
        chunks += [sentence[i:i + 50] for i in range(0,len(sentence),50)]

    return {'chunks':chunks}

chunked_dataset = dataset.map(chunk_examples,batched=True,remove_columns=dataset.column_names)
print(chunked_dataset[:10])
# 现在又更多的列了
print(dataset)
print(chunked_dataset)


"""
Data augmentation
"""
fillmask = pipeline('fill-mask',model='roberta-base')
mask_token = fillmask.tokenizer.mask_token
small_dataset = dataset.filter(lambda e,i:i<100,with_indices=True)

# 创造一个函数去随机的选择一个句子去mask在句子中，这个函数也返回原始的句子和由RoBERTA生成的两个替代
def augment_data(examples):
    outputs = []
    for sentence in examples['sentence1']:
        words = sentence.split(' ')
        K = randint(1,len(words) - 1)
        mask_sentence = " ".join(words[:K] + [mask_token] + words[K+1:])
        predictions = fillmask(mask_sentence)
        augment_sequences = [predictions[i]['sequence'] for i in range(3)]
        outputs +=[sentence] + augment_sequences

    return {'data':outputs}

augment_dataset = small_dataset.map(augment_data,batched=True,remove_columns=dataset.column_names,batch_size=8)
print(augment_dataset[:9]['data'])

# 同时处理多个分割
dataset = load_dataset('glue','mrpc')
encoded_dataset = dataset.map(lambda examples:tokenizer(examples['sentence1']),batched=True)
print(encoded_dataset["train"][0])

"""
分布式使用，就先算了这方面的
"""

"""
Concatenate
如果有相同的列类型，那么就可以连接起来
"""
bookcorpus = load_dataset("bookcorpus",split="train")
wiki = load_dataset("bookcorpus",split="train")
wiki = wiki.rename_columns("title")
assert bookcorpus.features.type == wiki.features.type
bert_dataset = concatenate_datasets([bookcorpus,wiki])

"""
interleaving：可以混合几个简单的数据集去从每个样本中交替抽取样本来创建一个新的数据集。
interleave_datasets和concatenate_datasets可以处理datasets.Dataset和datasets.IterableDataset类型上

"""

bookcorpus_ids = Dataset.from_dict({"ids":list(range(len(bookcorpus)))})
bookcorpus_with_ids = concatenate_datasets([bookcorpus,bookcorpus_ids],axis=1)


"""
设置数据的格式
"""
dataset.set_format(type='torch',columns=['input_ids','token_type_ids','attention_mask','label'])

# 使用下面的方法的话，不会对原始的数据操作，会返回一个新的
dataset = dataset.with_format(type='torch',columns=['input_ids','token_type_ids','attention_mask','label'])
# 使用reset方法可以把数据变回原来的格式
print(dataset.format)
dataset.reset_format()
print(dataset.format)

"""
Format transform可以即时的应用一个常用的格式转换
这会替代任何先前的具体的格式。
"""
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def encode(batch):
    return tokenizer(batch["sentence1"],padding="longest",truncation=True,max_length=512,return_tensors="pt")

print(dataset.set_transform(encode))
print(dataset.format)
print(dataset[:2])

"""
save
"""
encoded_dataset.save_to_disk("../dataset/")

# 重新加载数据
reloaded_encoded_dataset = load_from_disk("../dataset/directory")
