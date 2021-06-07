import numpy as np
dim = 1024
x = np.fromfile("wmt20-sent.en-ps.ps.emb", dtype=np.float32, count=-1)
x.resize(x.shape[0] // dim, dim)
y = np.fromfile("wmt20-sent.en-ps.en.emb", dtype=np.float32, count=-1)
y.resize(y.shape[0] // dim, dim)
from sklearn.metrics.pairwise import cosine_similarity
f = open("laser-scores","r")
sc = [float(i.rstrip().lstrip()) for i in f]
sc = np.nonzero(sc)[0][:1000]
print("here")
for i in sc:
    d = cosine_similarity(x[i].reshape(1,-1),y)[0]
    e = np.intersect1d(np.where(d > d[i]*1)[0],np.where(d < d[i]*1.01)[0])
    for item in e:
        print(str(i)+','+str(item))
print("here")
for i in sc:
    d = cosine_similarity(y[i].reshape(1,-1),x)[0]
    e = np.intersect1d(np.where(d > d[i]*1)[0],np.where(d < d[i]*1.01)[0])
    for item in e:
        print(str(i)+','+str(item))
