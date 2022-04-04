# HuggingFaceLearning
这个仓库用来存放我学习HuggingFace相关的代码和笔记



# Datasets

1. 对于比如GLUE数据集，一般都有几个子数据集，这些子数据集被叫做configurations
2. 



## question

1. 什么是Apache Arrow format？处理具有零拷贝读取的大型数据集，没有任何内存限制，以实现最佳速度和效率。





# Tokenizers

1. 在介绍主要特性的时候，说了一个Full alignment tracking，不太懂



### 3.Model

支持BPE、Unigram、WordLevel、WordPiece，更多细节，请点击[这里](https://huggingface.co/docs/tokenizers/python/latest/components.html#models)

WordPiece使用著名的## 前缀来识别作为单词一部分的标记（即不是单词的开头）。

BPE：最流行的子词标记化算法之一。 字节对编码的工作原理是从字符开始，同时将最常见的字符合并在一起，从而创建新的标记。 然后它迭代地工作以从它在语料库中看到的最常见的对中构建新的标记。

BPE 能够通过使用多个子词标记来构建它从未见过的词，因此需要更小的词汇表，具有“unk”（未知）标记的机会更少。



## question

1. 什么是BPE（Byte-Pair Encoding）？
2. 什么是attention mask？不是很理解。
3. 什么是legacy vocabulary files？
