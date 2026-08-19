import gensim
from gensim import corpora
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import warnings
from multiprocessing import freeze_support

warnings.filterwarnings('ignore')

# 读取数据
data_all = pd.read_csv("D:\\admin\\Desktop\\zw-24下-困惑度408.csv")
data = data_all['review_fine']
data = [eval(a) for a in data.values]

# 创建词典和语料库
dictionary = corpora.Dictionary(data)
corpus = [dictionary.doc2bow(text) for text in data]


def compute_metrics():
    """计算困惑度和一致性分数"""
    x = list(range(1, 30))
    z = []  # 困惑度
    y = []  # 一致性

    for num_topics in x:
        print(f"\n正在计算主题数: {num_topics}")

        # 训练LDA模型
        ldamodel = gensim.models.ldamodel.LdaModel(
            corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=30,
            random_state=1
        )

        # 计算困惑度
        perplex = ldamodel.log_perplexity(corpus)
        z.append(perplex)

        # 计算一致性
        coherence_model = gensim.models.coherencemodel.CoherenceModel(
            model=ldamodel,
            texts=data,
            dictionary=dictionary,
            coherence='c_v',
            processes=1  # 使用单进程
        )
        coherence_score = coherence_model.get_coherence()
        y.append(coherence_score)

        print(f"主题 {num_topics}: 困惑度={perplex:.4f}, 一致性={coherence_score:.4f}")

    return x, y, z


if __name__ == '__main__':
    freeze_support()

    x, y, z = compute_metrics()

    # 绘制图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(x, z, 'b-o')
    ax1.set_xlabel('主题数量')
    ax1.set_ylabel('困惑度分数')
    ax1.set_title('不同主题数量下的困惑度分数')
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, y, 'r-o')
    ax2.set_xlabel('主题数量')
    ax2.set_ylabel('一致性分数')
    ax2.set_title('不同主题数量下的一致性分数')
    ax2.grid(True, alpha=0.3)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    plt.tight_layout()
    plt.show()