import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

if __name__ == "__main__":
    with open('../data/train.csv', 'rb') as csvFile:
        titanic_reader = csv.reader(csvFile, delimiter=',', quotechar='"')
        row = titanic_reader.next()
        feature_names = np.array(row)

        titanic_X, titanic_y = [], []
        for row in titanic_reader:
            titanic_X.append(row)
            titanic_y.append(row[2])

        titanic_X = np.array(titanic_X)
        titanic_y = np.array(titanic_y)


    titanic_X = titanic_X[:, [2, 4, 5]]
    feature_names = feature_names[[2, 4, 5]]
    print feature_names

    ages = titanic_X[:, 2]
    mean_age = np.mean(titanic_X[ages != '', 2].astype(np.float))

    titanic_X[titanic_X[:, 2] == "NA", 2] = mean_age

    enc = LabelEncoder()
    label_encoder = enc.fit(titanic_X[:, 1])
    print "Categorical Classes : ", label_encoder.classes_

    integer_classes = label_encoder.transform(label_encoder.classes_)
    print "Integer Classes : ", integer_classes

    t = label_encoder.transform(titanic_X[:,1])
    titanic_X[:, 1] = t

    print feature_names
    print titanic_X[13], titanic_y[13]

    enc = LabelEncoder()
    label_encoder = enc.fit((titanic_X[:, 0]))
    print "Categorical Classes : ", label_encoder.classes_

    integer_classes = label_encoder.transform(label_encoder.classes_)
    print "Integer Classes : ", integer_classes

    enc = OneHotEncoder()
    one_hot_encoder = enc.fit(integer_classes)

    num_rows = titanic_X.shape[0]
    t = label_encoder.transform(titanic_X[:,0]).reshape(num_rows, 1)

    new_features = one_hot_encoder.transform(t)
    titanic_X = np.concatenate([titanic_X, new_features.toarray()], axis = 1)
    titanic_X = np.delete(titanic_X, [0], 1)

    feature_names = ['sex', 'age', 'first_class', 'second_class', 'third_class']
    titanic_X = titanic_X.astype(float)
    titanic_y = titanic_y.astype(float)

    print feature_names
    print titanic_X[0], titanic_y[0]