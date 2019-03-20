import pickle

data = pickle.load(open("data/sklearn-data.pickle", "rb"))
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

# Re-coding using HashingVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

vectorizer = HashingVectorizer(stop_words='english')
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.fit_transform(x_test)

# === Naive Bayes ===
# Fitting classifier to the training set
from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()
classifier.fit(x_train, y_train)

# Predict the test set results
y_pred = classifier.predict(x_test)

# Making the confusion matrix
from sklearn.metrics import accuracy_score

print("Naive Bayes' accuracy score: {}".format(accuracy_score(y_test, y_pred)))

# === Decision Tree ===
# Fitting classifier to the training set
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)
classifier.fit(x_train, y_train)

# Predict the test set results
y_pred = classifier.predict(x_test)

# Making the confusion matrix
from sklearn.metrics import accuracy_score

print("Decision Tree's accuracy score: {}".format(accuracy_score(y_test, y_pred)))
