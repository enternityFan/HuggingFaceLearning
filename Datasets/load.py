# @Time : 2022-04-05 7:45
# @Author : Phalange
# @File : load.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D
import pandas as pd
from datasets import load_dataset,load_metric
from datasets import Dataset,Features,Value,ClassLabel
import datasets
from datasets import ReadInstruction



# revision用来选择版本
dataset = load_dataset('lhoestq/demo1',revision='main',cache_dir="../cache/")
data_files = {"train":"train.csv","test":"test.csv"}
# 如果不具体data_files,将返回全部
dataset = load_dataset("namespace/your_dataset_name",data_files=data_files)

"""
加载CSV
"""

dataset = load_dataset('csv',data_files='my_file.csv')
# 假如要加载多个CSV
dataset = load_dataset('csv',data_files=['my_file_1.csv','my_file_2.csv','my_file_3.csv'])

# 还可以进行分割的功能
dataset = load_dataset('csv',data_files={'train':['my_train_file_1.csv','my_train_file_2.csv'],'test':'my_test_file.csv'})

# 加载远程CSV
base_url = "https://huggingface.co/datasets/lhoestq/demo1/resolve/main/data/"
dataset = load_dataset('csv',data_files={'train':base_url + 'train.csv','test':base_url + 'test.csv'})

# 加载zipped CSV files
url = "https://domain.org/train_data.zip"
data_files = {"train":url}
dataset = load_dataset("csv",data_files=data_files)

"""
加载JSON
JSON一般有多种格式，但我们认为最有效的格式是有多个JSON目标，每行都是一个object
"""

"""
下面演示的是这种数据格式的
{"a": 1, "b": 2.0, "c": "foo", "d": false}
{"a": 4, "b": -5.5, "c": null, "d": true}
"""
dataset = load_dataset('json',data_files='my_file.json')

"""
下面演示的是读取这种格式的JSON
{"version": "0.1.0",
    "data": [{"a": 1, "b": 2.0, "c": "foo", "d": false},
            {"a": 4, "b": -5.5, "c": null, "d": true}]
}
"""
dataset = load_dataset('json',data_files='my_file.json',field='data')

# 加载远程的json
base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
dataset = load_dataset('json', data_files={'train': base_url + 'train-v1.1.json', 'validation': base_url + 'dev-v1.1.json'}, field="data")


"""
加载Text文件
"""

dataset = load_dataset('text',data_files={'train':['my_text_1.txt','my_text_2.txt'],'test':'my_test_file.txt'})
# 加载远程的text
dataset = load_dataset('text', data_files='https://huggingface.co/datasets/lhoestq/test/resolve/main/some_text.txt')


"""
加载Parquet类型的数据
一般很大的数据集会用这种格式去存储，会更高效，并且更快返回查找值
"""
dataset = load_dataset('parquet',data_files={'train':'train.parquet','test':'test.parquet'})

# 加载远程的parquet文件
base_url = "https://storage.googleapis.com/huggingface-nlp/cache/datasets/wikipedia/20200501.en/1.0.0/"
data_files = {"train": base_url + "wikipedia-train.parquet"}
wiki = load_dataset("parquet", data_files=data_files, split="train")

"""
In-memory data 的读取也可以
通过创建datasets.Dataset来直接从python的dict格式或者Pandas.DataFrames里面读取数据
"""
my_dict = {"a":[1,2,3]}
dataset = Dataset.from_dict(my_dict)
df = pd.DataFrame({"a":[1,2,3]})
dataset = Dataset.from_pandas(df)

"""
OFFSET
当没用网络连接的时候，将会读取本地的缓存的数据集，
也可以通过设置 HF_DATASETS_OFFLINE变量为1去使用全离线模式
"""


"""
Slice splits
    一般有两个操作，使用string API或者是datasets.ReadInstruction
    Strings方法对于简单的案例来说更容易读
    datasets.ReadInstruction更容易使用各种slicing parameters

"""
"""
String API
"""
# 操作1.把train和test合并
train_test_ds = load_dataset('bookcorpus',split='train+test')
# 操作2.选择具体的行
train_10_20_ds = load_dataset('bookcorpus',split='train[10:20]')
train_10pct_ds = load_dataset('bookcorpus',split='train[:10%]')
train_10_80pct_ds = load_dataset('bookcorpus',split='train[:10%]+train[-80%:]')
# 操作3.创建一个交叉验证
# 10-fold cross-validation (see also next section on rounding behavior):
# The validation datasets are each going to be 10%:
# [0%:10%], [10%:20%], ..., [90%:100%].
# And the training datasets are each going to be the complementary 90%:
# [10%:100%] (for a corresponding validation set of [0%:10%]),
# [0%:10%] + [20%:100%] (for a validation set of [10%:20%]), ...,
# [0%:90%] (for a validation set of [90%:100%]).
vals_ds = load_dataset('bookcorpus', split=[f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)])
trains_ds = load_dataset('bookcorpus', split=[f'train[:{k}%]+train[{k+10}%:]' for k in range(0, 100, 10)])

"""
ReadInstruction的编写风格
"""

# 操作1.把train和test合并
ri = ReadInstruction('train') + ReadInstruction('test')
train_test_ds = load_dataset('bookcorpus',split=ri)
# 操作2.选择具体的行
train_10_20_ds = load_dataset('bookcorpus',split=datasets.ReadInstruction('train',to=10,unit='%'))
ri = (datasets.ReadInstruction('train', to=10, unit='%') + datasets.ReadInstruction('train', from_=-80, unit='%'))
train_10_80pct_ds = datasets.load_dataset('bookcorpus', split=ri)
# 操作3.创建一个交叉验证
# 10-fold cross-validation (see also next section on rounding behavior):
# The validation datasets are each going to be 10%:
# [0%:10%], [10%:20%], ..., [90%:100%].
# And the training datasets are each going to be the complementary 90%:
# [10%:100%] (for a corresponding validation set of [0%:10%]),
# [0%:10%] + [20%:100%] (for a validation set of [10%:20%]), ...,
# [0%:90%] (for a validation set of [90%:100%]).
vals_ds = datasets.load_dataset('bookcorpus', [datasets.ReadInstruction('train', from_=k, to=k+10, unit='%') for k in range(0, 100, 10)])
trains_ds = datasets.load_dataset('bookcorpus', [(datasets.ReadInstruction('train', to=k, unit='%') + datasets.ReadInstruction('train', from_=k+10, unit='%')) for k in range(0, 100, 10)])


"""
对于请求的切片边界没有被 100 整除的数据集，默认行为是将边界四舍五入到最接近的整数
"""
# Assuming *train* split contains 999 records.
# 19 records, from 500 (included) to 519 (excluded).
train_50_52_ds = datasets.load_dataset('bookcorpus', split='train[50%:52%]')
# 20 records, from 519 (included) to 539 (excluded).
train_52_54_ds = datasets.load_dataset('bookcorpus', split='train[52%:54%]')

# 如果想要均等分割，则使用pct1_dropremainder来做：
# 如果数据集中的示例数量没有除以 100，则使用 pct1_dropremainder 舍入可能会截断数据集中的最后一个示例。
# 18 records, from 450 (included) to 468 (excluded).
train_50_52pct1_ds = datasets.load_dataset('bookcorpus', split=datasets.ReadInstruction( 'train', from_=50, to=52, unit='%', rounding='pct1_dropremainder'))
# 18 records, from 468 (included) to 486 (excluded).
train_52_54pct1_ds = datasets.load_dataset('bookcorpus', split=datasets.ReadInstruction('train',from_=52, to=54, unit='%', rounding='pct1_dropremainder'))
# Or equivalently:
train_50_52pct1_ds = datasets.load_dataset('bookcorpus', split='train[50%:52%](pct1_dropremainder)')
train_52_54pct1_ds = datasets.load_dataset('bookcorpus', split='train[52%:54%](pct1_dropremainder)')


"""
Troubleshooting
主要是解决两个问题：手动下载数据集，并指定数据集的特征
    手动下载数据集有时会下载不下来，这个时候自己手动的打开链接下载，然后放到特定的位置上
"""
dataset = load_dataset("matinf", "summarization")
#Downloading and preparing dataset matinf/summarization (download: Unknown size, generated: 246.89 MiB, post-processed: Unknown size, total: 246.89 MiB) to /root/.cache/huggingface/datasets/matinf/summarization/1.0.0/82eee5e71c3ceaf20d909bca36ff237452b4e4ab195d3be7ee1c78b53e6f540e...
#AssertionError: The dataset matinf with config summarization requires manual data.
#Please follow the manual download instructions: To use MATINF you have to download it manually. Please fill this google form (https://forms.gle/nkH4LVE4iNQeDzsc9). You will receive a download link and a password once you complete the form. Please extract all files in one folder and load the dataset with: *datasets.load_dataset('matinf', data_dir='path/to/folder/folder_name')*.
#Manual data can be loaded with `datasets.load_dataset(matinf, data_dir='<path/to/manual/data>')


# Specify features
# 首先设置自己的特征
class_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
emotion_features = Features({'text': Value('string'), 'label': ClassLabel(names=class_names)})
# 然后进行设置
dataset = load_dataset('csv', data_files=file_dict, delimiter=';', column_names=['text', 'label'], features=emotion_features)

# 最后可以打印出来看看
print(dataset['train'].features)

"""
使用自己的Metrics
"""
metric = load_metric('PATH/TO/MY/METRIC/SCRPIT')

# Example of typical usage
for batch in dataset:
    inputs, references = batch
    predictions = model(inputs)
    metric.add_batch(predictions=predictions, references=references)
score = metric.compute()

# 一个metric可能有不同的配置，可以通过提供name的方式加载不同的
metric = load_metric('bleurt',name='bleurt-base-128')
metric = load_metric('bleurt',name='bleurt-base-512')

"""
分布式设置
1.num_process=总共的并行进程数目
2.process_id = 是进程的等级，从0到num_process-1
3.使用load_metric来加载这些参数
"""
#在后台datasets.Metric.compute()会聚集所有的结果得到最终的评价
metric = load_metric('glue','mrpc',num_process=num_process,process_id=rank)

# 有时，您可能在同一服务器和文件上同时运行多个独立的分布式评估。为了避免冲突，提供一个experiment_id去区分不同的结果是很重要的
metric = load_metric('glue','mrpc',num_process=num_process,process_id=process_id,experiment_id="My_experiment_10")





