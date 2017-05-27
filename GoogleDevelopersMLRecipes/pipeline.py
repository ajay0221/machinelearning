from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print "Decision Tree Accuracy ", accuracy_score(y_test, predictions)

from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier()

knn_classifier.fit(X_train, y_train)

knn_predictions = knn_classifier.predict(X_test)
print "KNN Classifier Accuracy ", accuracy_score(y_test, knn_predictions)