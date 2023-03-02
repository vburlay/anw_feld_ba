import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def source_date(path_to_file):
    data = pd.read_csv(path_to_file)
    return data


def train_test_dat(data):
    """

    :param data:
    :return: X_train, y_train, X_test, y_test
    """
    train = data[data['ORIGIN'] == 'train']
    y_train = train['CARAVAN']
    X_train = train.drop(['ORIGIN', 'CARAVAN'], axis=1)

    test = data[data['ORIGIN'] == 'test']
    y_test = test['CARAVAN']
    X_test = test.drop(['ORIGIN', 'CARAVAN'], axis=1)

    mm_scale = preprocessing.MinMaxScaler()
    X_train[X_train.columns] = mm_scale.fit_transform(X_train[X_train.columns])
    X_test[X_test.columns] = mm_scale.fit_transform(X_test[X_test.columns])

    return X_train, y_train, X_test, y_test


def pca_date(X_train, X_test):
    """

    :param X_train:
    :param X_test:
    :return: X_train_reduced, X_test_reduced
    """
    pca = PCA(n_components=0.95)
    X_train_reduced = pca.fit_transform(X_train)
    train_components = pca.n_components_

    X_test_reduced = pca.fit_transform(X_test)
    test_components = pca.n_components_

    X_test_reduced = X_test_reduced[:, 0:train_components]
    return X_train_reduced, X_test_reduced, train_components, test_components


def ml_model(X_train_reduced, X_test_reduced, X_train, X_test, y_train, y_test, train_components):
    log_reg = LogisticRegression(multi_class='ovr',
                                 class_weight=None,
                                 solver='saga',
                                 max_iter=10000)
    log_reg.fit(X_train_reduced, y_train)
    log_reg.predict(X_test_reduced)
    eval = log_reg.score(X_test_reduced, y_test)

    probs = log_reg.predict_proba(X_test_reduced)
    erg = probs[:, 1]

    return log_reg # Die Genauigkeit für den Testdatensatz


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')  # dashed diagonal
    plt.axis([0, 1, 0, 1])  # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)  # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)  # Not shown
    plt.grid(True)  # Not shown


def eval_train(log_reg,X_train_reduced,y_train):
    y_probas_log_reg = cross_val_predict(log_reg, X_train_reduced, y_train, cv=3, method="predict_proba")
    probs_lr = y_probas_log_reg[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, probs_lr)
    plt.figure(figsize=(8, 6))  # Not shown
    plot_roc_curve(fpr, tpr)
    plt.show()
