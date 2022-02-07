import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
# print("a")
#sns.set() # use seaborn plotting style
# print("a")
# Load the dataset
data = fetch_20newsgroups()
# print("a")
# Get the text categories
text_categories = data.target_names
# print("a")
# define the training set
train_data = fetch_20newsgroups(subset="train", categories=text_categories)

# print("a")
# define the test set
test_data = fetch_20newsgroups(subset="test", categories=text_categories)
# print("a")
print(text_categories)
print("We have "+ str(len(text_categories))+" unique categories")



print("We have {} training samples".format(len(train_data.data)))
print("We have {} test samples".format(len(test_data.data)))

# Build the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# Train the model using the training data
model.fit(train_data.data, train_data.target)

# Predict the categories of the test data
predicted_categories = model.predict(test_data.data)

# print(np.array(test_data.target_names)[predicted_categories])
# plot the confusion matrix
# mat = confusion_matrix(test_data.target, predicted_categories)
# sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=train_data.target_names,yticklabels=train_data.target_names)
# plt.xlabel("true labels")
# plt.ylabel("predicted label")
# plt.show()
print("The accuracy is "+str(accuracy_score(test_data.target, predicted_categories)))

def my_predictions(my_sentence, model):
    all_categories_names = np.array(data.target_names)
    prediction = model.predict([my_sentence])
    print(prediction)
    return all_categories_names[prediction]


my_sentence = """ That seemed pretty obvious even before he had kicked a ball in the competition. Getting the chance to play on the biggest stage in European club football was one of the main reasons Grealish gave for making the move to City from boyhood club Aston Villa for a British-record fee. His performance in the win over RB Leipzig only backed up the decision."""
print(my_predictions(my_sentence, model))
