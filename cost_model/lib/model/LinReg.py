from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

def get_metrics(y, y_preds):
    mae = mean_absolute_error(y, y_preds)
    r2 = r2_score(y, y_preds)
    metric_logs = {
        "MAE" : mae,
        "R2" : r2,
    }
    return metric_logs

def train(x, y):
    #reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
    reg = linear_model.LinearRegression()
    clf = reg.fit(x, y)
    y_preds = clf.predict(x)
    metric_logs = get_metrics(y, y_preds)
    print("Training Metrics : R2 {}; MAE {}".format(metric_logs["R2"], metric_logs["MAE"]))
    return clf

def test(clf, x, y):
    y_preds = clf.predict(x)
    metric_logs = get_metrics(y, y_preds)
    print("Testing Metrics : R2 {}; MAE {}".format(metric_logs["R2"], metric_logs["MAE"]))