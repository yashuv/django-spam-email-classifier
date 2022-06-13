import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


DATA_JSON_FILE = 'email-text-data.json'
data = pd.read_json(DATA_JSON_FILE)


vectorizer = CountVectorizer(stop_words='english')

all_features = vectorizer.fit_transform(data.MESSAGE)

X_train, X_test, y_train, y_test = train_test_split(all_features, data.CATEGORY, 
                                                   test_size=0.3, random_state=88)

classifier = MultinomialNB()        
classifier.fit(X_train, y_train)


msg = 'get viagra for free now!'

def email_prediction(msg):

    matrix = vectorizer.transform([msg])

    return classifier.predict(matrix)[0]

print(email_prediction(msg))