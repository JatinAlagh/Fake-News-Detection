import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# Loading And Preprocessing Dataset
news = pd.read_csv('news.csv', low_memory=False)

news['label'] = news['label'].map({'FAKE' : 0, 'REAL' : 1 })


news = news.drop(['title'],axis = 1 )
news = news.loc[:, ~news.columns.str.contains('^Unnamed')]

#Text Processing
def wordpot(text):
    # Check if the input is a string; if not, convert it to an empty string
    if isinstance(text, str):
        # converting into lower case
        text = text.lower()
        # removing urls
        text = re.sub(r'http?://\S+|www\.\S+', '', text)
        # removing tags
        text = re.sub(r'<.*?>', '', text)
        # removing punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # removing digits
        text = re.sub(r'\d', '', text)
        # removing newline characters
        text = re.sub(r'\n', '', text)
    else:
        text = ''

    return text


news["text"] = news["text"].apply(wordpot)
news = news.dropna(subset=['label', 'text'])

#Spliting The Dataset

x = news["text"]
y = news["label"]
x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.3)

#feature extraction and converting textual data into numerical
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
news = news.dropna(subset=['label'])
    

# Model Building

#logistic regression
from sklearn.linear_model import LogisticRegression
Lr = LogisticRegression()
Lr.fit(xv_train , y_train)
pred_lr = Lr.predict(xv_test)
# print(Lr.score(xv_test,y_test))
# print(classification_report(y_test,pred_lr))


# Manual Testing function

def output_label (n):
    if n == 0:
        return("It is a Fake news")
    elif n ==1:
        return("It is a Real news")

def manual_testing(news):
    testing_news ={"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test ["text"] = new_def_test["text"].apply(wordpot)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_lr = Lr.predict(new_xv_test)
    # pred_gbc = gbc.predict(new_xv_test)
    # pred_rfc = rfc.predict(new_xv_test)
    return "\n\nLr Prediction : {}".format(output_label(pred_lr[0]))

#Final prediction

news_article = str(input("Enter the news article for prediction: "))
result = manual_testing(news_article)
print(result)

