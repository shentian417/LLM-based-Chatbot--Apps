import ast
import warnings
import pandas as pd
import gensim
import gensim.corpora as corpora
from pprint import pprint

warnings.filterwarnings("ignore")

# ================= 只改这里 =================
INPUT_PATH = r"E:\\研一\\AA聊天机器人-2425年\\3-LDA\\中国\\24上\\zw-24上-困惑度.csv"
NUM_TOPICS = 11                  # 最佳主题数（困惑度算出）
OUT_PREFIX = "zw-24上"           # 输出文件名前缀，例如 zw-24上
USE_COLUMN = "review_fine"       # 中美统一用 review_fine
# ============================================

OUT_CSV = f"D:\\LDA-zw-24上-11.csv"
OUT_HTML = f"D:\\LDA-zw-24上-11.html"


def safe_parse(text):
    """把 CSV 里形如 ['很卡','闪退'] 的字符串还原成 list，空值/NaN 返回 []"""
    if text is None:
        return []
    if not isinstance(text, str):
        if pd.isna(text):
            return []
    s = str(text).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]
    except (ValueError, SyntaxError):
        cleaned = s.strip("[]'\"")
        return cleaned.split() if cleaned else []
    except Exception:
        return []


# 加载数据集（显式 utf-8-sig，去掉 BOM）
data_all = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
train_fine_text = data_all[USE_COLUMN].apply(safe_parse).tolist()
nonempty = sum(1 for d in train_fine_text if d)
print(f"总文档数: {len(train_fine_text)}，非空文档: {nonempty}")

# 构建词典和语料
id2word = corpora.Dictionary(train_fine_text)
corpus = [id2word.doc2bow(t) for t in train_fine_text]
print(f"词典大小: {len(id2word)}")

# 构建 LDA 模型
lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=NUM_TOPICS,
    random_state=100,
    alpha='asymmetric',
    eta='auto',
    passes=30,
    update_every=0,
    iterations=500,
    gamma_threshold=0.001,
)

print("Alpha参数:", lda_model.alpha)
print("Eta参数:", lda_model.eta)
print("\n各主题关键词:")
pprint(lda_model.print_topics(num_topics=NUM_TOPICS, num_words=30))

# 每篇文档的主题概率（min=0 返回全部分布）
doc_topic = [lda_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]

topic_columns = [f"Topic {i}" for i in range(NUM_TOPICS)]
topic_data = [[dict(doc).get(i, 0.0) for i in range(NUM_TOPICS)] for doc in doc_topic]
topic_score = pd.DataFrame(topic_data, columns=topic_columns)

# 主导主题：直接复用上面已算好的分布，避免二次计算口径不一致
dominant_topics = []
for d in doc_topic:
    if d:
        tid, prob = max(d, key=lambda x: x[1])
        dominant_topics.append((tid, prob))
    else:
        dominant_topics.append((-1, 0.0))  # 空文档标记为 -1

topic_score['Dominant_Topic'] = [t for t, _ in dominant_topics]
topic_score['Topic_Probability'] = [p for _, p in dominant_topics]

# 结果导出（utf-8-sig，Excel 不乱码）
result = pd.concat([data_all.reset_index(drop=True), topic_score], axis=1)
result.to_csv(OUT_CSV, encoding="utf-8-sig", index=False)
print(f"结果已保存至: {OUT_CSV}")

# LDAvis 可视化
try:
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
    vis_data = gensimvis.prepare(
        lda_model, corpus, id2word,
        mds='tsne',
        R=30,
        plot_opts={'xlab': 't-SNE1', 'ylab': 't-SNE2'}
    )
    pyLDAvis.save_html(vis_data, OUT_HTML)
    print(f"可视化已保存至: {OUT_HTML}")
except Exception as e:
    print(f"可视化失败（不影响主结果）: {e}")
