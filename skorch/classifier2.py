import skorch
import numpy as np
from sklearn.datasets import make_classification
from torch import nn
from skorch import NeuralNetClassifier
import pandas as pd
import pickle
import torch
import argparse
import torch

#Neural Net architecture

class Net(nn.Module):
    def __init__(self,input_shape):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_shape,2048)
        self.fc2 = nn.Linear(2048,2048)
        self.fc3 = nn.Linear(2048,2048)
        self.fc4 = nn.Linear(2048,2048)
        self.fc5 = nn.Linear(2048,2048)
        self.fc6 = nn.Linear(2048,2)
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x):
        # x = np.concatenate((endict[int(x.split('|')[0])],psdict[int(x.split('|')[1])]))
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout(x)
        x = torch.relu(self.fc5(x))
        x = self.dropout(x)
        x = torch.relu(self.fc6(x))
        x = self.softmax(x)
        return x


#parsing all the arguments

parser = argparse.ArgumentParser()
parser.add_argument("--starting_rank", required=True,help="starting rank for the classifier", type=int)
parser.add_argument("--ending_rank", required=True,help="ending rank for the classifier, make sure it's not more than 1000+starting rank", type=int)
parser.add_argument("--model_name", default="model", help="model name where model will be saved, assumed this is consistent for language pairs")
parser.add_argument("--previous_scores", default="wmt20-sent.en-ps.laser-score", help="file where scores of the previous iteration are saved")
parser.add_argument("--current_scores", default="scores.txt", help="file where scores of the current iteration are saved")
parser.add_argument("--src_embedding", default="wmt20-sent.en-ps.ps.xlmr", help="source embedding to use for training. Note: Change loading based on encoder")
parser.add_argument("--tgt_embedding", default="wmt20-sent.en-ps.en.xlmr", help="target embedding to use for training. Note: Change loading based on encoder")
parser.add_argument("--neg_src", default="neg-1.txt", help="file with negative samples generated using source language")
parser.add_argument("--neg_tgt", default="neg-2.txt", help="file with negative samples generated using target language")
parser.add_argument("--iteration", default=1, help="iteration number", type=int)
args = parser.parse_args()


#reading embeddings
f = open(args.tgt_embedding,"r")
endict = [np.array(list(map(float,i.lstrip().rstrip().split(' ')))) for i in f]
f.close()
f = open(args.src_embedding,"r")
psdict = [np.array(list(map(float,i.lstrip().rstrip().split(' ')))) for i in f]
f.close()

#get previous iteration's scores along with the positive samples's indexes
f = open(args.previous_scores,"r")
sc = [float(i.rstrip().lstrip()) for i in f]
a = sc
a = sorted(range(len(a)), key=lambda i: a[i], reverse=True)[args.starting_rank:args.ending_rank]
sc = a

X7 = [str(i)+'|'+str(i) for i in sc]
y7 = [1 for i in range(len(X7))]
f.close()


#get the negative samples associated with the positive samples
f = open(args.neg_src,"r")
X1 = X1 = [str(i.split(',')[1].lstrip().rstrip())+'|'+str(i.split(',')[0].rstrip().lstrip()) for i in f]
y1 = [0 for i in range(len(X1))]
f.close()
print("here2")

f = open(args.neg_tgt,"r")
X2 = [str(i.split(',')[0].lstrip())+'|'+str(i.split(',')[1].rstrip()) for i in f]
y2 = [0 for i in range(len(X2))]
f.close()
print("here3")


X3 = [str(i-1)+'|'+str(i-1) for i in sc]
X4 = [str(i+1)+'|'+str(i+1) for i in sc]
y3 = [0 for i in range(len(X3))]
y4 = [0 for i in range(len(X4))]


#combine everything and balance the classes
X = X1 + X2 + X3 + X4
y = y1 + y2 + y3 + y4
t = int(len(X)/len(X7))


for i in range(t):
    X = X + X7
    y = y + y7
    

#initializing the model and training parameters
model = NeuralNetClassifier(
    Net(len(np.concatenate((endict[int(X[0].split('|')[0])],psdict[int(X[0].split('|')[1])])))),
    max_epochs=10,
	batch_size = 10,
	optimizer=torch.optim.SGD,
	optimizer__momentum=0.95,
    lr=0.01,
    device='cuda',
    iterator_train__shuffle=True,
	warm_start=True,
    train_split = None
)

#training the model
for i in range(0,len(X)):
    X_t = torch.FloatTensor(np.concatenate((endict[int(X[i].split('|')[0])],psdict[int(X[i].split('|')[1])]), axis = None).reshape(1,-1))
    y_t = torch.tensor(np.array(y[i]).reshape(1,-1))
    model.fit(X_t,y_t[0])

try:
    pickle.dump(model, open(args.model_name,"wb"))
except:
    c = 0

f = open(args.current_scores,"w")
for i in range(len(psdict)):
    try:
        t = torch.FloatTensor(np.concatenate((endict[i],psdict[i]), axis = None).reshape(1,-1))
        f.write(str(model.predict_proba(np.array(t).reshape(1,-1))[0][1]))
        f.write("\n")
    except:
        f.write("0")
        f.write("\n")
