import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

dir = os.path.dirname(__file__)
filepath = 'production/HealthData.csv'

def GetStandardScalarForHeart():
    dataset = pd.read_csv(os.path.join(dir, filepath))
    x = dataset.iloc[:, :13].values
    y = dataset.iloc[:, 13].values

    sc = SimpleImputer(missing_values=np.nan, strategy='mean')
    sc.fit(x[:, 11:13])
    x[:, 11:13] = sc.transform(x[:, 11:13])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    s = StandardScaler()
    x_train = s.fit_transform(x_train)
    x_test = s.transform(x_test)

    return s,x_train, x_test, y_train, y_test


def GetSVCClassifierForHeart(x_train,y_train,x_test,y_test):
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)

    accuracy = accuracy_score(y_test,y_pred)
    accuracy = int(accuracy*10000)
    accuracy = accuracy/100
    return classifier, accuracy

def GetLogisticRegressionClassifierForHeart(x_train, y_train,x_test,y_test):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = int(accuracy * 10000)
    accuracy = accuracy / 100
    return classifier, accuracy

def GetNaiveBayesClassifierForHeart(x_train,y_train,x_test,y_test):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = int(accuracy * 10000)
    accuracy = accuracy / 100
    return classifier, accuracy

def GetDecisionTreeClassifierForHeart(x_train, y_train,x_test,y_test):
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = int(accuracy * 10000)
    accuracy = accuracy / 100
    return classifier, accuracy

def GetKNNClassifierForHeart(x_train, y_train,x_test,y_test):
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5,metric="minkowski", p=2)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = int(accuracy * 10000)
    accuracy = accuracy / 100
    return classifier, accuracy