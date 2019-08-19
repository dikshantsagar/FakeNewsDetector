import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

train=pd.read_csv('dataset/train2.tsv', sep='\t',header=None)

traindata=[]
for i in range(len(train)):
    data=train.iloc[:,3][i]+" "+str(train.iloc[:,4][i])+" "+str(train.iloc[:,5][i])+" "+str(train.iloc[:,6][i])+" "+str(train.iloc[:,14][i])
    label=train.iloc[:,2][i]
    traindata.append([data,label])

train=pd.DataFrame(traindata)
print("Training Data Fetched -------------")
labdict={'barely-true':0,
         'false':1,
         'half-true':2,
         'mostly-true':3,
         'pants-fire':4,
         'true':5}

for i in range(len(train)):
    train.iloc[:,1][i]=labdict[train.iloc[:,1][i]]

train=train.replace('\d+', 'NUM', regex=True)

cv=TfidfVectorizer(ngram_range=(1,3), max_features=10000)
x=cv.fit_transform(train.iloc[:,0])

clf=LinearSVC(C=0.1)
clf.fit(x,list(train.iloc[:,1]))
print("Model Trained -----------------")
test=pd.read_csv('dataset/test2.tsv', sep='\t',header=None)
testdata=[]
for i in range(len(test)):
    data=test.iloc[:,3][i]+" "+str(test.iloc[:,4][i])+" "+str(test.iloc[:,5][i])+" "+str(test.iloc[:,6][i])+" "+str(test.iloc[:,14][i])
    label=test.iloc[:,2][i]
    testdata.append([data,label])

test=pd.DataFrame(testdata)
print("Testing Data Fetched -------------")
for i in range(len(test)):
    test.iloc[:,1][i]=labdict[test.iloc[:,1][i]]

test=test.replace('\d+', 'NUM', regex=True)

xt=cv.transform(test.iloc[:,0])

print("Accuracy Achieved :",clf.score(xt,list(test.iloc[:,1])))

