from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import pickle
import torch

f = open("wmt20-sent.en-ps.en.xlmr","r")
endict = [np.array(list(map(float,i.lstrip().rstrip().split(' ')))) for i in f]
f.close()
f = open("wmt20-sent.en-ps.ps.xlmr","r")
psdict = [np.array(list(map(float,i.lstrip().rstrip().split(' ')))) for i in f]
f.close()
print("here")
f = open("scores.txt","r")
sc = [float(i.rstrip().lstrip()) for i in f]
sc = np.nonzero(sc)[0][:1000]
X7 = [list(psdict[int(i)])+list(endict[int(i)]) for i in sc]
y7 = [1 for i in range(len(X7))]
f.close()
print("here1")
f = open("mono-1.txt","r")
X1 = [list(psdict[int(i.split(',')[1].lstrip())])+list(endict[int(i.split(',')[0].rstrip())]) for i in f]
y1 = [0 for i in range(len(X1))]
f.close()
print("here2")
f = open("mono-2.txt","r")
X2 = [list(psdict[int(i.split(',')[0].lstrip())])+list(endict[int(i.split(',')[1].rstrip())]) for i in f]
y2 = [0 for i in range(len(X2))]
f.close()
print("here3")

X3 = [list(psdict[i-1])+list(endict[i-1]) for i in sc]
X4 = [list(psdict[i+1])+list(endict[i+1]) for i in sc]
y3 = [0 for i in range(len(X3))]
y4 = [0 for i in range(len(X4))]

#f = open("neg-1.txt","r")
#X3 = [list(psdict[int(i.split(',')[1].lstrip())])+list(endict[int(i.split(',')[0].rstrip())]) for i in f]
#y3 = [0 for i in range(len(X3))]
#f.close()
#print("here4")
#f = open("neg-2.txt","r")
#X4 = [list(psdict[int(i.split(',')[0].lstrip())])+list(endict[int(i.split(',')[1].rstrip())]) for i in f]
#y4 = [0 for i in range(len(X4))]
#f.close()
#print("here5")
#f = open("pos-1.txt","r")
#X5 = [list(psdict[int(i.split(',')[1].lstrip())])+list(endict[int(i.split(',')[0].rstrip())]) for i in f]
#y5 = [0 for i in range(len(X5))]
#f.close()
#print("here6")
#f = open("pos-2.txt","r")
#X6 = [list(psdict[int(i.split(',')[0].lstrip())])+list(endict[int(i.split(',')[1].rstrip())]) for i in f]
#y6 = [0 for i in range(len(X6))]




X = X1 + X2 + X3 + X4
y = y1 + y2 + y3 + y4
t = int(len(X)/len(X7))
for i in range(t):
    X = X + X7
    y = y + y7
for i in range(len(X)):
 if len(X[i]) != 2048:
  del X[i]
  del y[i]
clf = MLPClassifier(random_state=1, verbose = True, early_stopping=True, learning_rate="invscaling", learning_rate_init=0.000002, hidden_layer_sizes=(2048,2048,2048,2048,)).fit(X, y)
try:
    pickle.dump(clf, open("model4","wb"))
except:
    c = 0
f = open("/home/ankur/laser/LASER/tasks/given-23/scores.txt","w")
for i in range(len(psdict)):
    t = list(psdict[int(i)])+list(endict[int(i)])
    if len(t) != 2048:
        f.write("0")
        f.write("\n")
    else:
        f.write(str(clf.predict_proba(np.array(t).reshape(1,-1))[0][1]))
        f.write("\n")
