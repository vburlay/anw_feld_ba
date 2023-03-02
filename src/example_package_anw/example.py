import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  cross_val_predict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import seaborn as sns
from sklearn import metrics

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
    y_pred_df = pd.DataFrame(probs)
    y_pred_1 = y_pred_df.iloc[:, [1]]

    y_test_df = pd.DataFrame(y_test)

    # Put the index as ID column, remove index from both dataframes and combine them
    y_test_df["ID"] = y_test_df.index
    y_pred_1.reset_index(drop=True, inplace=True)
    y_test_df.reset_index(drop=True, inplace=True)
    y_pred_final = pd.concat([y_test_df, y_pred_1], axis=1)
    y_pred_final = y_pred_final.rename(columns={1: "Yes_Prob", "CARAVAN": "Yes"})
    y_pred_final = y_pred_final.reindex(["ID", "Yes", "Yes_Prob"], axis=1)

    # Create columns with different probability cutoffs
    numbers = [float(x) / 10 for x in range(10)]
    for i in numbers:
        y_pred_final[i] = y_pred_final.Yes_Prob.map(lambda x: 1 if x > i else 0)

    # Calculate accuracy, sensitivity & specificity for different cut off points
    Probability = pd.DataFrame(columns=['Probability', 'Accuracy', 'Sensitivity', 'Specificity'])

    for i in numbers:
        CM = metrics.confusion_matrix(y_pred_final.Yes, y_pred_final[i])
        Total = sum(sum(CM))
        Accuracy = (CM[0, 0] + CM[1, 1]) / Total
        Sensitivity = CM[1, 1] / (CM[1, 1] + CM[1, 0])
        Specificity = CM[0, 0] / (CM[0, 0] + CM[0, 1])
        Probability.loc[i] = [i, Accuracy, Sensitivity, Specificity]
    Probability.plot.line(x='Probability', y=['Accuracy', 'Sensitivity', 'Specificity'])

    y_pred_final['predicted'] = y_pred_final.Yes_Prob.map(lambda x: 1 if x > 0.1 else 0)
    confusion_matrix = metrics.confusion_matrix(y_pred_final.Yes, y_pred_final.predicted)
    Probability[Probability["Probability"] == 0.1]

    return log_reg # Die Genauigkeit für den Testdatensatz




def roc(log_reg,X_train_reduced,y_train):
    y_probas_log_reg = cross_val_predict(log_reg, X_train_reduced, y_train, cv=3, method="predict_proba")
    probs_lr = y_probas_log_reg[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, probs_lr)
    plt.plot(fpr, tpr, linewidth=2, label=None)
    plt.plot([0, 1], [0, 1], 'k--')  # dashed diagonal
    plt.axis([0, 1, 0, 1])  # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)  # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)  # Not shown
    plt.grid(True)  # Not shown
    plt.show()
    plt.figure(figsize=(8, 6))  # Not shown
    plt.show()


def cor(X_train_reduced):
    # Check the correlations between components
    corr_mat = np.corrcoef(X_train_reduced.transpose())
    plt.figure(figsize=[15, 8])
    sns.heatmap(corr_mat)
    plt.show()
