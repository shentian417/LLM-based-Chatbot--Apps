import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import re
import jieba
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel


# 加载数据集
data_all = pd.read_csv("C:\\Users\\Shen Tian\\Desktop\\聊天机器人分词.csv")

# 去除文本中的表情符号（只保留中文和数字）
def clear_character(sentence):
    pattern = re.compile('[^\u4e00-\u9fa5^a-z^A-Z^0-9]')
    line = re.sub(pattern,'',sentence)
    new_sentence = ''.join(line.split())
    return new_sentence
train_text = [clear_character(data_all) for data_all in data_all['review']]
data_all['review_clear'] = train_text

# 添加词典
jieba.load_userdict("C:\\Users\\Shen Tian\\Desktop\\jiebaDic.txt")

# 分词
train_seg_text = [jieba.lcut(s) for s in train_text]
data_all['review_seg'] = train_seg_text

# 加载停用词
stop_words_path = "C:\\Users\\Shen Tian\\Desktop\\stop word.txt"

def get_stop_words():
    return set([item.strip() for item in open(stop_words_path,'r').readlines()])

stopwords = get_stop_words()

# 去除文本中的停用词
def drop_stopwords(line):
    line_clear = []
    for word in line:
        if word in stopwords:
            continue
        line_clear.append(word)
    return line_clear

train_st_text = [drop_stopwords(s) for s in train_seg_text]
data_all['review_st'] = train_st_text

def is_fine_word(words, min_length=2):
    line_clear = []
    rule = re.compile(r"^[\u4e00-\u9fa5]+$")
    for word in words:
        if len(word) >= min_length and re.search(rule, word):
            line_clear.append(word)
    return line_clear
train_fine_text = [is_fine_word(s,min_length=2) for s in train_st_text]
data_all['review_fine'] = train_fine_text

bigram = gensim.models.Phrases(train_fine_text,min_count=5,threshold=5)
trigram = gensim.models.Phrases(bigram[train_fine_text],threshold=5)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def make_bigram(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigram(texts):
    return [trigram_mod[doc] for doc in texts]

data_words_bigrams = make_bigram(train_fine_text)
data_words_trigrams = make_trigram(train_fine_text)

data_all['bigrams'] = data_words_bigrams
data_all['trigrams'] = data_words_trigrams
# 数据预处理的结果输出
data_all.to_csv('C:\\Users\\Shen Tian\\Desktop\\聊天机器人分词-result.csv',encoding="utf-8")






