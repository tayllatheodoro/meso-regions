from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


def random_forest_classifier(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred


def svm_classifier(X_train, y_train, X_test, y_test):
    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred


def logistic_regression_classifier(X_train, y_train, X_test, y_test):
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred


def xgboost_classifier(X_train, y_train, X_test, y_test):
    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred
