# @Time : 2022-04-04 11:05
# @Author : Phalange
# @File : BERTtokenizer.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase,NFD,StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from tokenizers import decoders
bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
bert_tokenizer.normalizer = normalizers.Sequence([NFD(),Lowercase()])
bert_tokenizer.pre_tokenizer = Whitespace()

bert_tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair = "[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]",1),
        ("[SEP]",2),
    ],
)

trainer = WordPieceTrainer(
    vocab_size=30522,special_tokens=["[UNK]","[CLS]","[SEP]","[PAD]","[MASK]"])

files = [f"../dataset/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
bert_tokenizer.train(files, trainer)

bert_tokenizer.save("../dataset/bert-wiki.json")

"""
Decoding
    把生成的ID转换为文本
"""

output = bert_tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.ids)

print(bert_tokenizer.decode(output.ids))

#如果您使用的模型添加了特殊字符来表示给定“单词”的子标记（如 WordPiece 中的“##”），则需要自定义解码器以正确处理它们。
# 比如
output = bert_tokenizer.encode("Welcome to the 🤗 Tokenizers library.")
print(output.tokens)
# ["[CLS]", "welcome", "to", "the", "[UNK]", "tok", "##eni", "##zer", "##s", "library", ".", "[SEP]"]

bert_tokenizer.decode(output.ids)
# "welcome to the tok ##eni ##zer ##s library ."



# 通过下面方式解决
bert_tokenizer.decoder = decoders.WordPiece()
bert_tokenizer.decode(output.ids)
# "welcome to the tokenizers library."
