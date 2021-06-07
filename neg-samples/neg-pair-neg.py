
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument("--src-embedding", required=False, default="wmt20-sent.en-ps.ps.emb", help="file with the source side embeddings")
parser.add_argument("--tgt-embedding", required=False, default="wmt20-sent.en-ps.en.emb", help="file with the target side embeddings")
parser.add_argument("--src-nnegs", required=False, default="neg-1.txt", help="file with the source side based negative samples on the negative side")
parser.add_argument("--tgt-nnegs", required=False, default="neg-2.txt", help="file with the target side based negative samples on the negative side")
parser.add_argument("--src-pnegs", required=False, default="pos-1.txt", help="file with the source side based negative samples on the positive side")
parser.add_argument("--tgt-pnegs", required=False, default="pos-2.txt", help="file with the target side based negative samples on the positive side")
parser.add_argument("--limit", required=False, default=0.01, type=int)
parser.add_argument("--laser-scores", required=False, default="laser-scores", help="file with the target side embeddings")
args = parser.parse_args()

#reading embeddings
dim = 1024
x = np.fromfile(args.src-embedding, dtype=np.float32, count=-1)
x.resize(x.shape[0] // dim, dim)
y = np.fromfile(args.tgt-embedding, dtype=np.float32, count=-1)
y.resize(y.shape[0] // dim, dim)
f = open(args.laser-scores,"r")
sc = [float(i.rstrip().lstrip()) for i in f]
sc = np.nonzero(sc)[0][:1000]


#using embeddings to get cosine distances
limit = args.limit
f1 = open(args.src-nnegs,"w")
f2 = open(args.src-pnegs,"w")
for i in sc:
    d = cosine_similarity(x[i].reshape(1,-1),y)[0]
    e = np.intersect1d(np.where(d < d[i]*1)[0],np.where(d > d[i]*(1-limit))[0])
    for item in e:
        f1.write(str(i)+','+str(item)+'\n')
    e = np.intersect1d(np.where(d > d[i]*1)[0],np.where(d > d[i]*(1+limit))[0])
    for item in e:
        f2.write(str(i)+','+str(item)+'\n')

f1 = open(args.tgt-nnegs,"w")
f2 = open(args.tgt-pnegs,"w")
for i in sc:
    d = cosine_similarity(y[i].reshape(1,-1),x)[0]
    e = np.intersect1d(np.where(d < d[i]*1)[0],np.where(d > d[i]*(1-limit))[0])
    for item in e:
        f1.write(str(i)+','+str(item))
    e = np.intersect1d(np.where(d > d[i]*1)[0],np.where(d > d[i]*(1+limit))[0])
    for item in e:
        f2.write(str(i)+','+str(item)+'\n')
