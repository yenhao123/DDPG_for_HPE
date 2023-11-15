from sklearn import linear_model
import numpy as np

def train(x, y):
    reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
    print(x)
    print(y)
    clf = reg.fit(x, y)
    return clf

def test(clf, x, y):
    y_preds = clf.predict(x)
    print(y)
    print(y_preds)
    mse = 1 / y.shape[0] * np.sum(np.power(y - y_preds, 2))
    print(mse)