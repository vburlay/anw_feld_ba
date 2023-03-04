import numpy as np
from sklearn.model_selection import cross_val_score


class model():
    def __init__(self):
        self.model_abrv = {'dt': 'Decision Tree Classifier',
                           'rf': 'Random Forest Classifier',
                           'svc': 'Support Vector Machines',
                           'lr': 'Logistic Regression'}
        self.res_ev = dict()

    def model_ev(self):
        """Trains models and outputs score metrics. Takes an identifier, list of models, and split dataset as inputs and has options for saving model,
        printing confusion matrix and classification report and getting cross-validated 5 fold accuracy."""
        clf_model = self.models[self.clf]
        clf_model.fit(self.X_train, self.y_train)
        clf_model.predict(self.X_test)
        self.res_ev[self.model_abrv[self.clf]] = np.mean(
            cross_val_score(clf_model, self.X_train, self.y_train, cv=5, scoring='accuracy'))
