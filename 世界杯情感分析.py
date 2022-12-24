import pandas as pd
import jieba
import numpy as np
from PIL import Image
from os import path


df = pd.read_excel(r"D:\Rayi\大学\大学学习\大三课程\数据挖掘\世界杯\虎扑评论.xlsx")
words= []
for i,row in df.iterrows():
    word = jieba.cut(row['content'])
    result = ' '.join(word)
    words.append(result)
print(words)
pl = '评论'
for i in range(len(words)):
    pl = pl + words[i]

from imageio import imread
import jieba.analyse
import wordcloud
from imageio import imread
d=path.dirname(r"D:\Rayi\大学\大学学习\大三课程\数据挖掘\世界杯\mask-1.png")
mask = np.array(Image.open(path.join(d, "mask-1.png")))
excludes = {}
jieba.analyse.set_stop_words(r"D:\Rayi\大学\大学学习\大三课程\数据挖掘\实验5\stopwords.txt")
ls = jieba.analyse.extract_tags(pl, topK=100)
print(ls)
txt = " ".join(ls)
w = wordcloud.WordCloud(width = 1000,height = 700, background_color = "white" , mask =mask, font_path='simhei.ttf')
w.generate(txt)
w.to_file("wordcloud.png")

def drop_stopwords(contents,stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords or not '\u4e00' <= word <= '\u9fff':
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean,all_words

import pandas as pd
from snownlp import SnowNLP
import matplotlib. pyplot as plt

#打开待分析虎扑文本
df = pd.read_excel(r"D:\Rayi\大学\大学学习\大三课程\数据挖掘\世界杯\虎扑评论.xlsx")
#定义函数，批量处理所有的评论信息
def get_sentiment_hupu(text):
    s = SnowNLP(text)
    return s.sentiments
df ["sentiment"] = df.content.apply(get_sentiment_hupu)
print(df)
x=df ["Num"]
y=df ["sentiment"]
plt. figure(figsize=(20,8), dpi=80)
plt.plot(x,y)
plt. savefig('./虎扑情感分析.jpg' )