from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

#load and split iris data into the features and labeled sets
data = load_iris()
features = data["data"]
labels = data["target"]

#STEP 1: Split the data so 80% goes into training and 20% goes into testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20)

#initialize and fit the model to the data
model = DecisionTreeClassifier()
model.fit(features_train, labels_train)

#test the mdoel
predictions = model.predict(features_test)

#STEP 2: measure the accuracy in two different ways
accuracy = accuracy_score(labels_test, predictions)
accuracy2 = model.score(features_test, labels_test)
print("The mean accuracy of the model is {}.".format(accuracy2))


#STEP 3: cross validate and check which max_depth is the best for this model. This will likely help avoid overfitting.
max_depths = [1, 5, 10]
best_depth = {"depth": [],
              "accuracy": -1}

for max_depth in max_depths:
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(features_train, labels_train)
    accuracy2 = model.score(features_test, labels_test)
    if accuracy2 == best_depth["accuracy"]:
        best_depth["depth"].append(max_depth)
    if accuracy2 > best_depth["accuracy"]:
        best_depth["depth"] = [max_depth]
        best_depth["accuracy"] = accuracy2
        
if len(best_depth["depth"]) > 1:
    print("The best max depths for the model are {}, each with an accuracy of {}".format(best_depth["depth"], best_depth["accuracy"]))
else:
    print("The best max depth for the model is {}, with an accuracy of {}".format(best_depth["depth"][0], best_depth["accuracy"]))


#STEP 4: create a confusion matrix with results from step 3, and answer which class has the most false positives (in output)

#Calculate Confusion Matrix
model = DecisionTreeClassifier(max_depth=best_depth["depth"][0])
model.fit(features_train, labels_train)
predictions = model.predict(features_test)
cm = confusion_matrix(labels_test, predictions, labels = [0,1,2])
print(cm)

#calculate False Positives
most_false_positives = {"label": [],
                        "fps": -1}
for i in range(len(cm[0])):
    fps = sum(cm[:,i]) - cm[i][i]
    if fps == most_false_positives["fps"]:
        most_false_positives["label"].append(i)
        most_false_positives["fps"] = fps
    if fps > most_false_positives["fps"]:
        most_false_positives["label"] = [i]
        most_false_positives["fps"] = fps

if len(most_false_positives["label"]) > 1:
    print("The labels {}, each have {} False Positives".format(most_false_positives["label"],most_false_positives["fps"]))
else:
    print("The label {} has {} False Positives".format(most_false_positives["label"][0], most_false_positives["fps"]))

        
        