# @Time : 2022-04-04 14:55
# @Author : Phalange
# @File : QuickStart.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm

"""
把微调的BERT用于释义分类任务的全过程熟悉
"""
#train_dataset = load_dataset("ag_news", split="train[:40000]",verify=False)
#print("dataset loading success!")
dataset = load_dataset("glue",'mrpc',split='train',cache_dir="../cache/")

model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
def encode(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')

dataset = dataset.map(encode, batched=True)
print(dataset[0])


# 1.将标签列重命名为标签，即 BertForSequenceClassification 中的预期输入名称
dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

# 2.从 Dataset 对象中检索实际张量，而不是使用当前 Python 对象。


# 3.过滤数据集以仅返回模型输入：input_ids、token_type_ids 和 attention_mask。

# datasets.Dataset.set_format() 实现了上面说的2 3两步
dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
print(next(iter(dataloader)))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.train().to(device)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
for epoch in range(3):
    for i, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            print(f"loss: {loss}")