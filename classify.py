# All the required imports
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.stats import loguniform

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV


def train_classifier(X, y):
    """Train a classifier using the given training data.

    Trains a logistic regression on the input data with default parameters.
    """
    print("Started Training the Classifier")
    cls = LogisticRegression(solver="liblinear", penalty="l2", C=2.1212)
    cls.fit(X, y)
    '''
    space = dict()
    space['solver'] = ['liblinear']
    space['penalty'] = ['l2']
    space['C'] = loguniform(1e-5, 100)
    # define search
    search = RandomizedSearchCV(
        cls, space, n_iter=50, scoring='accuracy', random_state=21, verbose=4)
    # execute search
    result = search.fit(X, y)

    # summarize result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    '''
    return cls


def evaluate(X, yt, cls):
    """Evaluated a classifier on the given labeled data using accuracy."""
    print("Started Evaluating the Classifier")
    yp = cls.predict(X)
    acc = metrics.accuracy_score(yt, yp)
    print("Accuracy :", acc)
    return acc
