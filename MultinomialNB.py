import pandas as pd
from nltk.corpus import stopwords
import string
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from model import MultiNB

file_path = "./spam.csv"
df = pd.read_csv(file_path, encoding='iso8859_14')
df.drop(labels=df.columns[2:],axis=1,inplace=True)
df.columns=['target','text']

def clean_util(text):
    punc_rmv = [char for char in text if char not in string.punctuation]
    punc_rmv = "".join(punc_rmv)
    stopword_rmv = [w.strip().lower() for w in punc_rmv.split() if w.strip().lower() not in stopwords.words('english')]
    return " ".join(stopword_rmv)

df['text'] = df['text'].apply(clean_util)


cv = CountVectorizer()
X = cv.fit_transform(df['text']).toarray()
lb = LabelBinarizer()
y = lb.fit_transform(df['target']).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = MultiNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(model.score(X_test, y_test))

