from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

#load and split iris data into the features and labeled sets
data = load_iris()
features = data["data"]
labels = data["target"]

#STEP 1: Split the data so 80% goes into training and 20% goes into testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2)

#initialize and fit the model to the data
model = DecisionTreeClassifier()
model.fit(features_train, labels_train)

#test the mdoel
predictions = model.predict(features_test)

#Step2: measure the accuracy in two different ways
accuracy = accuracy_score(labels_test, predictions)
accuracy2 = model.score(features_test, labels_test)
print("The mean accuracy of the model is {}.".format(accuracy))