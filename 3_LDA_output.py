# 第三步 构建LDA模型，输出文档-主题对应关系，并进行可视化
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import re
import jieba
import gensim
from pprint import pprint
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim
import chardet  # 用于检测文件编码

# 加载数据集
print("正在加载数据...")
try:
    data_all = pd.read_csv("D:\\admin\\Desktop\\zw-24下-困惑度408.csv")
    print(f"数据加载完成，共 {len(data_all)} 条记录")
except Exception as e:
    print(f"加载数据失败: {e}")
    exit()


# 去除文本中的表情符号（只保留中英文和数字）
def clear_character(sentence):
    pattern = re.compile('[^\u4e00-\u9fa5^a-z^A-Z^0-9]')
    line = re.sub(pattern, '', sentence)
    new_sentence = ''.join(line.split())
    return new_sentence


print("清洗文本...")
train_text = [clear_character(text) for text in data_all['review']]

# 加载自定义词典
print("加载自定义词典...")
try:
    jieba.load_userdict("D:\\admin\\Desktop\\jiebaDic.txt")
    print("自定义词典加载成功")
except Exception as e:
    print(f"自定义词典加载失败: {e}")

# 分词
print("进行分词...")
train_seg_text = [jieba.lcut(s) for s in train_text]
print(f"分词完成，前5个文档的分词示例:")
for i in range(min(5, len(train_seg_text))):
    print(f"  文档{i + 1}: {train_seg_text[i][:10]}...")  # 显示前10个词

# 加载停用词
print("加载停用词...")
stop_words_path = "D:\\admin\\Desktop\\stop word.txt"


def get_stop_words():
    try:
        # 尝试检测文件编码
        with open(stop_words_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            print(f"检测到停用词文件编码: {encoding}")

        # 用检测到的编码读取文件
        with open(stop_words_path, 'r', encoding=encoding, errors='ignore') as f:
            words = [item.strip() for item in f.readlines() if item.strip()]
        return set(words)
    except UnicodeDecodeError:
        # 如果检测失败，尝试常见编码
        encodings = ['gbk', 'gb2312', 'utf-8', 'latin1']
        for enc in encodings:
            try:
                with open(stop_words_path, 'r', encoding=enc, errors='ignore') as f:
                    words = [item.strip() for item in f.readlines() if item.strip()]
                print(f"使用编码 {enc} 成功读取停用词文件")
                return set(words)
            except:
                continue
        # 如果所有编码都失败，返回空集
        print("警告: 无法读取停用词文件，将使用空停用词列表")
        return set()
    except Exception as e:
        print(f"读取停用词文件失败: {e}")
        return set()


stopwords = get_stop_words()
print(f"加载了 {len(stopwords)} 个停用词")


# 去除停用词
def drop_stopwords(line):
    return [word for word in line if word not in stopwords]


train_st_text = [drop_stopwords(s) for s in train_seg_text]
print(f"去除停用词后，前5个文档示例:")
for i in range(min(5, len(train_st_text))):
    print(f"  文档{i + 1}: {train_st_text[i][:10]}...")


# 过滤词长（只保留长度≥2的中文）
def is_fine_word(words, min_length=2):
    line_clear = []
    rule = re.compile(r"^[\u4e00-\u9fa5]+$")
    for word in words:
        if len(word) >= min_length and re.search(rule, word):
            line_clear.append(word)
    return line_clear


train_fine_text = [is_fine_word(s, min_length=2) for s in train_st_text]
print(f"过滤后，前5个文档示例:")
for i in range(min(5, len(train_fine_text))):
    print(f"  文档{i + 1}: {train_fine_text[i][:10]}...")

total_words = sum(len(doc) for doc in train_fine_text)
print(f"文本处理完成，总词数: {total_words}")

# 构建bigram和trigram
print("构建bigram和trigram...")
try:
    bigram = gensim.models.Phrases(train_fine_text, min_count=5, threshold=5)
    trigram = gensim.models.Phrases(bigram[train_fine_text], threshold=5)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)


    def make_bigram(texts):
        return [bigram_mod[doc] for doc in texts]


    def make_trigram(texts):
        return [trigram_mod[doc] for doc in texts]


    data_words_bigrams = make_bigram(train_fine_text)
    data_words_trigrams = make_trigram(train_fine_text)
    processed_texts = data_words_trigrams
    print("bigram和trigram构建成功")
except Exception as e:
    print(f"bigram/trigram构建失败，使用原始文本: {e}")
    processed_texts = train_fine_text

print(f"处理后的文本示例: {processed_texts[0][:10]}...")

# 构建词典和语料向量化
print("构建词典和语料...")
id2word = corpora.Dictionary(processed_texts)
corpus = [id2word.doc2bow(text) for text in processed_texts]
print(f"词典大小: {len(id2word)}")
print(f"语料库大小: {len(corpus)}")

# 构建LDA模型
print("开始训练LDA模型（主题数=12）...")
try:
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=12,
        random_state=100,
        update_every=0,
        chunksize=1000,
        passes=30,
        eta='auto',
        alpha='asymmetric',
        per_word_topics=True
    )
    print("LDA模型训练完成！")
except Exception as e:
    print(f"LDA模型训练失败: {e}")
    exit()

# 打印模型参数
print("\n" + "=" * 50)
print("模型参数:")
print(f"Alpha参数: {lda_model.alpha}")
print(f"Eta参数: {lda_model.eta}")

# 打印每个主题的关键词
print("\n" + "=" * 50)
print("各主题关键词（前30个）:")
pprint(lda_model.print_topics(num_topics=12, num_words=30))

# 获取每篇文档的主题概率
print("\n" + "=" * 50)
print("计算文档主题概率...")
doc_topic = []
for i, bow in enumerate(corpus):
    try:
        topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
        topic_dict = dict(topic_dist)
        doc_topic.append(topic_dict)
    except Exception as e:
        print(f"文档 {i} 主题概率计算失败: {e}")
        doc_topic.append({})

# 构建主题概率DataFrame
topic_columns = [f"Topic_{i}" for i in range(12)]
topic_data = []
for doc in doc_topic:
    topic_probs = [doc.get(topic_id, 0.0) for topic_id in range(12)]
    topic_data.append(topic_probs)

topic_score = pd.DataFrame(topic_data, columns=topic_columns)

# 获取主导主题
dominant_topics = []
for i, doc in enumerate(doc_topic):
    if doc:
        try:
            dominant_topic_id = max(doc.items(), key=lambda x: x[1])[0]
            dominant_prob = doc[dominant_topic_id]
            dominant_topics.append((dominant_topic_id, dominant_prob))
        except:
            dominant_topics.append((-1, 0.0))
    else:
        dominant_topics.append((-1, 0.0))

# 添加主导主题和主导概率
topic_score['Dominant_Topic'] = [topic_id for topic_id, _ in dominant_topics]
topic_score['Dominant_Probability'] = [prob for _, prob in dominant_topics]

# 合并原始数据和主题概率
print("生成结果文件...")
try:
    result = pd.concat([data_all.reset_index(drop=True), topic_score], axis=1)

    # 保存CSV文件
    output_path = "D:\\admin\\Desktop\\LDA-zw\\zw-24下-12_documents_topics.csv"
    result.to_csv(output_path, encoding='utf-8-sig', index=False)  # 使用utf-8-sig支持Excel
    print(f"文档-主题对应关系已保存至: {output_path}")
    print(f"文件包含 {len(result)} 行，{len(result.columns)} 列")

    # 显示前几行数据
    print("\n前3行数据预览:")
    print(result[['review'] + topic_columns[:3] + ['Dominant_Topic', 'Dominant_Probability']].head(3))
except Exception as e:
    print(f"保存结果文件失败: {e}")

# 显示主导主题分布
print("\n" + "=" * 50)
print("主导主题分布:")
dominant_counts = result['Dominant_Topic'].value_counts().sort_index()
for topic, count in dominant_counts.items():
    if topic != -1:
        percentage = (count / len(result)) * 100
        print(f"主题 {topic}: {count} 个文档 ({percentage:.2f}%)")

# ===================== LDA可视化 =====================
print("\n" + "=" * 50)
print("生成LDA可视化...")
try:
    # 准备可视化数据
    vis_data = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

    # 保存为HTML文件
    viz_path = "D:\\admin\\Desktop\\LDA-zw\\zw-24下-12_visualization.html"
    pyLDAvis.save_html(vis_data, viz_path)
    print(f"LDA可视化已保存至: {viz_path}")
    print("提示: 在浏览器中打开此文件查看交互式可视化图表")
except Exception as e:
    print(f"LDA可视化生成失败: {e}")
    print("可能是pyLDAvis版本问题，可以尝试安装离线版本: pip install pyLDAvis[offline]")

# 额外生成一个主题关键词摘要
print("\n" + "=" * 50)
print("主题关键词摘要（每个主题前10个关键词）:")
for topic_id in range(13):
    try:
        topic_words = lda_model.show_topic(topic_id, topn=10)
        words = [word for word, _ in topic_words]
        print(f"主题 {topic_id}: {', '.join(words)}")
    except:
        print(f"主题 {topic_id}: 无法获取关键词")

print("\n" + "=" * 50)
print("处理完成！")
if 'output_path' in locals():
    print(f"1. 文档-主题对应关系: {output_path}")
if 'viz_path' in locals():
    print(f"2. LDA交互式可视化: {viz_path}")