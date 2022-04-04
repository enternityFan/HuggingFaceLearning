# @Time : 2022-04-04 10:21
# @Author : Phalange
# @File : tokenizationPipeline.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers.normalizers import NFD,StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Digits
from tokenizers.processors import TemplateProcessing


"""
当使用encode或者encode_batch时，输入的文本一般经过下面流水线
1.正则化
2.预处理化
3.模型
4.后处理
"""

tokenizer = Tokenizer.from_file("../dataset/tokenizer-wiki.json")

#----------------------------------------------------------------------------------
"""
1.正则化
"""




# 应用UFD编码正则化，并且移出读音
normalizer = normalizers.Sequence([NFD(),StripAccents()])
# 如果用下式改变正则化方法，你也需要重新训练一下tokenizer
tokenizer.normalizer = normalizer # 调整tokenizer的正则化属性
print(normalizer.normalize_str("Héllò hôw are ü?"))

"""
2.Pre-Tokenization
"""
pre_tokenizer = Whitespace()
output = pre_tokenizer.pre_tokenize_str("Hello!How are you?i'm fine,thank you.")# 分割发音也会分割i'm
print(output) # 输出是一个列表，每个元素是一个二元组（词，（起始位置，终止位置））

# 结合任意的PreTokenizer在一起
pre_tokenizer = pre_tokenizers.Sequence([Whitespace(),Digits(individual_digits=True)])
output = pre_tokenizer.pre_tokenize_str("Call 911!")
print(output)

# 同样的方式，可以通过下面的方法来改变一个Tokenizer
tokenizer.pre_tokenizer = pre_tokenizer
# 当然，这样的话就需要重新训练咯

"""
3.模型
    1>模型的作用是使用它学习到的规则去分词
    2>负责把token映射到模型词汇表中它对应的ID
    一般支持：BPE、Unigram、WordLevel、WordPiece这几个库
    
    这个就不说了吧，在前面tokenize.py中有做
"""



"""
4.post-processing
    这个与正则化和预处理词元不同的时，经过下面的方法指定属性后，不需要重新训练tokenizer

"""
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
)



