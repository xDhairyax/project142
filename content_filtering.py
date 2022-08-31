from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

df=pd.read_csv("articles.csv")
df=df[df['soup'].notna()]

count=CountVectorizer(stop_words="english")
countmatrix=count.fit_transform(df["soup"])

cosine_sim2=cosine_similarity(countmatrix,countmatrix)

df=df.reset_index()
indices=pd.Series(df.index,index=df["title"])
def get_recommendations(title,cosine_sim):
  idx=indices[title]
  simsscore=list(enumerate(cosine_sim[idx]))
  simsscore=sorted(simsscore,key=lambda x:x[1],reverse=True)
  simsscore=simsscore[1:11]
  articleindices=[i[0]for i in simsscore]
  return df["title"].iloc[articleindices]