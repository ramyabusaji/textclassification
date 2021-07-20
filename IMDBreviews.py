import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split as tts

data = pd.read_csv('Dataset.csv',encoding='latin-1')

data.head()

data.shape

data.columns

data['sentiment'].value_counts()

def clean_text1(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text=re.sub('\w*\d\w*','',text)
    return text

cleaned1=lambda x:clean_text1(x)

data['review']=pd.DataFrame(data.review.apply(cleaned1))

data.head()

def clean_text2(text):
    text=re.sub('[''"",,,]','',text)
    text=re.sub('\n','',text)
    return text

cleaned2=lambda x:clean_text2(x)

data['review']=pd.DataFrame(data.review.apply(cleaned2))
data.head()

x = data.iloc[0:,0].values
y = data.iloc[0:,1].values

xtrain,xtest,ytrain,ytest = tts(x,y,test_size = 0.25,random_state = 225)

tf = TfidfVectorizer()
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
model=Pipeline([('vectorizer',tf),('classifier',classifier)])

model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

accuracy_score(ypred,ytest)*100

A=confusion_matrix(ytest,ypred)
print(A)

recall=A[0][0]/(A[0][0]+A[1][0])
precision=A[0][0]/(A[0][0]+A[0][1])
F1=2*recall*precision/(recall+precision)
print(F1)

pre = model.predict(["Production has an incredibly important place to shoot a series or film. Sometimes even a very minimalist story can reach an incredibly successful point after the right production stages. The Witcher series is far from minimalist. The Witcher is one of the best Middle-earth works in the world. Production quality is essential if you want to handle such a topic successfully."])
print(f'Prediction: {pre[0]}')
#Prediction: positive

pre = model.predict(["I think this is my first review. This series is so bad I had to write one. I don't understand the good score. I have tried on 2 separate occasions to watch this show. Haven't even gotten past the 2nd episode because it is SO BORING."])
print(f'Prediction: {pre[0]}')
#Prediction: negative


pre=model.predict(["movie was awesome"])
print(f'Prediction: {pre[0]}')
#Prediction: positive

pre=model.predict(["worst movie"])
print(f'Prediction: {pre[0]}')
#Prediction: negative



