from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import pickle
import torch
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser()
parser.add_argument("--starting_rank", required=True,help="starting rank for the classifier", type=int)
parser.add_argument("--ending_rank", required=True,help="ending rank for the classifier, make sure it's not more than 1000+starting rank", type=int)
parser.add_argument("--model_name", default="model", help="model name where model will be saved, assumed this is consistent for language pairs")
parser.add_argument("--previous_scores", default="wmt20-sent.en-ps.laser-score", help="file where scores of the previous iteration are saved")
parser.add_argument("--current_scores", default="scores.txt", help="file where scores of the current iteration are saved")
parser.add_argument("--src_embedding", default="wmt20-sent.en-ps.ps.xlmr", help="source embedding to use for training. Note: Change loading based on encoder")
parser.add_argument("--tgt_embedding", default="wmt20-sent.en-ps.en.xlmr", help="target embedding to use for training. Note: Change loading based on encoder")
parser.add_argument("--neg_src", default="mono-1.txt", help="file with negative samples generated using source language")
parser.add_argument("--neg_tgt", default="mono-2.txt", help="file with negative samples generated using target language")
args = parser.parse_args()



class Net(nn.Module):
    def __init__(self,input_shape):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_shape,2048)
        self.fc2 = nn.Linear(2048,2048)
        self.fc3 = nn.Linear(2048,2048)
        self.fc4 = nn.Linear(2048,2048)
        self.fc5 = nn.Linear(2048,2048)
        self.fc6 = nn.Linear(2048,1)
        self.dropout = nn.Dropout(p=0.1)
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.dropout(x)
        return x


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc



class trainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__ (self):
        return len(self.X_data)



#load embeddings stored in theform of a dictionary
f = open(args.tgt_embedding,"r")
endict = [np.array(list(map(float,i.lstrip().rstrip().split(' ')))) for i in f]
f.close()
f = open(args.src_embedding,"r")
psdict = [np.array(list(map(float,i.lstrip().rstrip().split(' ')))) for i in f]
f.close()


#get previous iteration's scores along with the positive samples's indexes
f = open(args.previous_scores,"r")
sc = [float(i.rstrip().lstrip()) for i in f]
sc = np.nonzero(sc)[0][args.starting_rank:args.ending_rank]

X7 = [list(psdict[int(i)])+list(endict[int(i)]) for i in sc]
y7 = [1 for i in range(len(X7))]
f.close()



#get the negative samples associated with the positive samples
f = open(args.neg_src,"r")
X1 = [list(psdict[int(i.split(',')[1].lstrip())])+list(endict[int(i.split(',')[0].rstrip())]) for i in f]
y1 = [0 for i in range(len(X1))]
f.close()
print("here2")

f = open(args.neg_tgt,"r")
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



#combine the positive and negative samples to get training data and make sure that we don't have sentences with an invalid embedding (due to a max size limit on xlmr)
X = X1 + X2 + X3 + X4
y = y1 + y2 + y3 + y4
t = int(len(X)/len(X7))


for i in range(t):
    X = X + X7
    y = y + y7
for i in range(len(X)):
# print(i)
    if len(X[i]) != 2048:
        del X[i]
        del y[i]



#actually do the training and save the results
#try:
#    clf = pickle.load(open(args.model_name, 'rb'))
#    clf.fit(X,y)
#except:
#    clf = MLPClassifier(random_state=1, verbose = True, early_stopping=False, warm_start = True, learning_rate="invscaling", learning_rate_init=0.00002, max_iter=1000, hidden_layer_sizes=(2048,2048,2048,2048,)).fit(X, y)
#try:
#    pickle.dump(clf, open(args.model_name,"wb"))
#except:
#    c = 0

learning_rate = 0.01
epochs = 700
BATCH_SIZE = 10

model = Net(input_shape=2048)
model.cuda()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
loss_fn = nn.BCEWithLogitsLoss()

losses = []
accur = []


train_data = trainData(torch.FloatTensor(X), torch.FloatTensor(y))
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

for i in range(epochs):
    for j,(X,y) in enumerate(train_loader):
        #calculate output
        output = model(X[0].cuda())
        #calculate loss
        loss = loss_fn(output,y.reshape(-1,1))
        #accuracy
        predicted = model(torch.tensor(X,dtype=torch.float32))
        acc = (predicted.reshape(-1).detach().numpy().round() == y).mean()
        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if i%50 == 0:
        losses.append(loss)
        accur.append(acc)
        print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,acc))


f = open(args.current_scores,"w")
for i in range(len(psdict)):
    t = list(psdict[int(i)])+list(endict[int(i)])
    if len(t) != 2048:
        f.write("0")
        f.write("\n")
    else:
        f.write(str(clf.predict_proba(np.array(t).reshape(1,-1))[0][1]))
        f.write("\n")
