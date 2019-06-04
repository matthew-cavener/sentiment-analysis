import utils

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

data = utils.read_data()
X = utils.combined_embeddings(data.text.tolist())
y = data.sentiment


def hyperopt_train_SVC(params):
    clf = SVC(decision_function_shape='ovo', probability=True, **params)
    return cross_val_score(clf, X, y).mean()

space = {
    'C': hp.uniform('C', 0, 10),
    'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'rbf']),
    'gamma': hp.uniform('gamma', 0, 10)
}

def f(params):
    acc = hyperopt_train_SVC(params)
    return {
        'loss': -acc,
        'status': STATUS_OK
    }

trials = Trials()
best = fmin(
    f,
    space,
    algo=tpe.suggest,
    max_evals=1000,
    trials=trials
)
print('best:')
print(best)
