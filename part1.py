import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


train=pd.read_csv('dataset/train2.tsv', sep='\t',header=None)

traindata=[]
for i in range(len(train)):
    data=train.iloc[:,3][i]+" "+str(train.iloc[:,4][i])+" "+str(train.iloc[:,5][i])+" "+str(train.iloc[:,6][i])+" "+str(train.iloc[:,14][i])
    label=train.iloc[:,2][i]
    traindata.append([data,label])

train=pd.DataFrame(traindata)
print("Training Data Fetched -------------")
for i in range(len(train)):
    if(train.iloc[:,1][i]=='true' or train.iloc[:,1][i]=='mostly-true' or train.iloc[:,1][i]=='half-true' ):
        train.iloc[:,1][i]=1
    else:
        train.iloc[:,1][i]=0

train=train.replace('\d+', 'NUM', regex=True)

cv=CountVectorizer(max_features=10000,ngram_range=(1,3))

x=cv.fit_transform(train.iloc[:,0])
clf=MultinomialNB(alpha=4)
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
    if(test.iloc[:,1][i]=='true' or test.iloc[:,1][i]=='mostly-true' or test.iloc[:,1][i]=='half-true'):
        test.iloc[:,1][i]=1
    else:
        test.iloc[:,1][i]=0

test=test.replace('\d+', 'NUM', regex=True)

xt=cv.transform(test.iloc[:,0])

print("Accuracy Achieved :",clf.score(xt,list(test.iloc[:,1])))