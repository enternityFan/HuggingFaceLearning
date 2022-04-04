# @Time : 2022-04-04 21:45
# @Author : Phalange
# @File : metric.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

from datasets import list_metrics
from datasets import load_metric
import datasets
metrics_list = list_metrics()
print(len(metrics_list))

print(metrics_list)

metric = load_metric('glue', 'mrpc')
print(metric.inputs_description)

glue_metric = datasets.load_metric('glue', 'sst2')  # 'sst2' or any of ["mnli", "mnli_mismatched", "mnli_matched", "qnli", "rte", "wnli", "hans"]
references = [0, 1]
predictions = [0, 1]
results = glue_metric.compute(predictions=predictions, references=references)
print(results)
glue_metric = datasets.load_metric('glue', 'mrpc')  # 'mrpc' or 'qqp'
references = [0, 1]
predictions = [0, 1]
results = glue_metric.compute(predictions=predictions, references=references)
print(results)

# 下面两行就是评估预测的方法
#model_predictions = model(model_inputs)
#final_score = metric.compute(predictions=model_predictions, references=gold_references)