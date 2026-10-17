import warnings
warnings.filterwarnings("ignore")

import os
import re
import pandas as pd
import gensim

from nltk.tokenize import RegexpTokenizer

# 英文分词器：按纯字母切词（无需下载 punkt 等额外数据）
_EN_TOKENIZER = RegexpTokenizer(r"[a-z]+")

STOPWORDS_PATH = r"C:\Users\Shen Tian\Desktop\聊天机器人代码\stop word yw.txt"

# (输入, 输出) —— 美国 4 个时期的英文评论
US_FILES = [
    (r"E:\研一\AA聊天机器人-2425年\3-LDA\美国\24上\yw-24上-分词.csv",
     r"E:\研一\AA聊天机器人-2425年\3-LDA\美国\24上\yw-24上-分词-result.csv"),
    (r"E:\研一\AA聊天机器人-2425年\3-LDA\美国\24下\yw-24下-分词.csv",
     r"E:\研一\AA聊天机器人-2425年\3-LDA\美国\24下\yw-24下-分词-result.csv"),
    (r"E:\研一\AA聊天机器人-2425年\3-LDA\美国\25上\yw-25上-分词.csv",
     r"E:\研一\AA聊天机器人-2425年\3-LDA\美国\25上\yw-25上-分词-result.csv"),
    (r"E:\研一\AA聊天机器人-2425年\3-LDA\美国\25下\yw-25下-分词.csv",
     r"E:\研一\AA聊天机器人-2425年\3-LDA\美国\25下\yw-25下-分词-result.csv"),
]


def load_stopwords(path):
    """读取英文停用词表，统一转小写"""
    words = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                words.add(w)
    return words


def clear_character(sentence):
    """清洗：只保留英文字母、数字、空格，转小写（保留空格以分隔单词）"""
    if not isinstance(sentence, str):
        return ""
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", sentence.lower())
    return re.sub(r"\s+", " ", s).strip()


def tokenize_en(sentence):
    """小写 + NLTK 英文分词（RegexpTokenizer）"""
    if not isinstance(sentence, str):
        return []
    return _EN_TOKENIZER.tokenize(sentence.lower())


def drop_stopwords(tokens, stopwords):
    return [w for w in tokens if w not in stopwords]


def is_fine_word(tokens, min_length=2):
    """只保留纯英文单词，长度 >= min_length"""
    rule = re.compile(r"^[a-z]+$")
    return [w for w in tokens if len(w) >= min_length and rule.match(w)]


def process_one(input_path, output_path, stopwords):
    print(f"处理中：{os.path.basename(input_path)}")
    data = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"  共 {len(data)} 条评论")

    # 1. 清洗
    data["review_clear"] = [clear_character(t) for t in data["review"]]

    # 2. 分词（NLTK）
    data["review_seg"] = [tokenize_en(t) for t in data["review"]]

    # 3. 去停用词
    data["review_st"] = [drop_stopwords(t, stopwords) for t in data["review_seg"]]

    # 4. 过滤：只保留纯英文、长度>=2
    data["review_fine"] = [is_fine_word(t) for t in data["review_st"]]

    # 5. bigram / trigram
    try:
        bigram = gensim.models.Phrases(data["review_fine"], min_count=5, threshold=5)
        trigram = gensim.models.Phrases(bigram[data["review_fine"]], threshold=5)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        data["bigrams"] = [bigram_mod[doc] for doc in data["review_fine"]]
        data["trigrams"] = [trigram_mod[doc] for doc in data["review_fine"]]
    except Exception as e:
        print(f"  bigram/trigram 构建失败，跳过：{e}")
        data["bigrams"] = data["review_fine"]
        data["trigrams"] = data["review_fine"]

    data.to_csv(output_path, encoding="utf-8-sig", index=False)
    print(f"  已保存：{output_path}\n")


def main():
    stopwords = load_stopwords(STOPWORDS_PATH)
    print(f"加载英文停用词 {len(stopwords)} 个\n")
    for inp, out in US_FILES:
        process_one(inp, out, stopwords)
    print("全部 4 个时期处理完成！")


if __name__ == "__main__":
    main()
