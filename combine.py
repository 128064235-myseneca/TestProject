import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def classify():


    data = fetch_20newsgroups()
    text_categories = data.target_names
    train_data = fetch_20newsgroups(subset="train", categories=text_categories)
    test_data = fetch_20newsgroups(subset="test", categories=text_categories)

    # print("We have {} unique classes".format(len(text_categories)))
    # print("We have {} training samples".format(len(train_data.data)))
    # print("We have {} test samples".format(len(test_data.data)))

    model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    model.fit(train_data.data, train_data.target)

    predicted_categories = model.predict(test_data.data)



    print("The accuracy of text classification is {}".format(accuracy_score(test_data.target, predicted_categories)))
    return (data, model)
def my_predictions(my_sentence, model, data):
    all_categories_names = np.array(data.target_names)
    prediction = model.predict([my_sentence])
    pred = all_categories_names[prediction][0]
    lst = pred.split(".")
    return lst[1]

def preprocess_data():
    sdata = pd.read_csv('google_play_store_apps_reviews_training.csv')

    sdata = sdata.drop('package_name', axis=1)

    sdata['review'] = sdata['review'].str.strip().str.lower()
    return sdata

def sentiment(sdata):

    x = sdata['review']
    y = sdata['polarity']
    x, x_test, y, y_test = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)


    vec = CountVectorizer(stop_words='english')
    x = vec.fit_transform(x).toarray()
    x_test = vec.transform(x_test).toarray()



    smodel = MultinomialNB()
    smodel.fit(x, y)

    print("The accuracy of semantic analysis is ", smodel.score(x_test, y_test))
    return smodel, vec

data, model = classify()
my_sentence = """ That seemed pretty obvious even before he had kicked a ball in the competition. Getting the chance to play on the biggest stage in European club football was one of the main reasons Grealish gave for making the move to City from boyhood club Aston Villa for a British-record fee. His performance in the win over RB Leipzig only backed up the decision."""
print("Input: ", my_sentence)
print("The prediction is ", my_predictions(my_sentence, model, data))


print("\n")
sdata = preprocess_data()
smodel, vec = sentiment(sdata)
inp = 'The way this app is made is really good'
print("Input: ", inp)
print("The prediction is ", smodel.predict(vec.transform([inp])))
#
# from flask import Flask
#
# app = Flask(__name__)
#
# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"
#
#
# if __name__ == '__main__':
#    app.run()