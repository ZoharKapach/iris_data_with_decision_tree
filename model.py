from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

#load and split iris data into the features and labeled sets
data = load_iris()
features = data["data"]
labels = data["target"]

#initialize and fit the model to the data
model = DecisionTreeClassifier()
model.fit(features, labels)

#test the mdoel
predictions = model.predict(features)

print(predictions)