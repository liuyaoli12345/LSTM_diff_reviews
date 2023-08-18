from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import pandas as pd

def get_vector(text):
    vectors = []
    words = text.split()
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
    return vectors
# ##可以用下面的方式将词语‘computer’变成向量，可以 print(vector)试试看是什么
# vector = model.wv['computer']
# print(vector)

data = pd.read_csv("IMDB Dataset.csv")

##下面这句话是构建 Word2Vec 模型，自行查看官方文档，了解每一个参数的含义 
model = Word2Vec(sentences=data['review'], vector_size=100, window=5, min_count=1, workers=4)
##然后保存模型 
model.save("word2vec.model")

vector_label = pd.DataFrame(columns=['vector','label'])
vector_label['vector'] = data['review'].apply(get_vector)
vector_label['label'] = data['sentiment'].map({'positive': 1, 'negative': 0})
vector_label.to_csv("vector_label.csv", index=False)

