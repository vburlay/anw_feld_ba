from keras.models import Sequential
import keras
from keras.layers import Dense
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold


class model_cnn:
    def __init__(self, x_traincnn, x_testcnn, y_train, y_test):
        self.x_traincnn = x_traincnn
        self.x_testcnn = x_testcnn
        self.y_train = y_train.astype('float32')
        self.y_test = y_test.astype('float32')
        # Merge inputs and targets
        self.inputs = np.concatenate((x_traincnn, x_testcnn), axis=0)
        self.targets = np.concatenate((y_train, y_test), axis=0)
        # Define per-fold score containers
        self.acc_per_fold = []
        self.loss_per_fold = []
        # Define the K-fold Cross Validator
        self.kfold = KFold(n_splits=5, shuffle=True)

    def cnn(self):
        # K-fold Cross Validation model evaluation
        fold_no = 1
        for train, test in self.kfold.split(self.inputs, self.targets):
            # Define the model architecture
            model = Sequential()

            model.add(Conv1D(128, 5, padding='same',
                             input_shape=(85, 1)))
            model.add(Activation('relu'))
            model.add(Dropout(0.1))
            model.add(MaxPooling1D(pool_size=(8)))
            model.add(Conv1D(128, 5, padding='same', ))
            model.add(Activation('relu'))
            model.add(Dropout(0.1))
            model.add(Flatten())
            model.add(Dense(2))  # auf 2 setzen und neu trainieren
            model.add(Activation('softmax'))
            opt = keras.optimizers.RMSprop(lr=0.00005, rho=0.9, epsilon=None, decay=0.0)

            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=opt,
                          metrics=["acc"])
            # Generate a print
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')
            self.cnnhistory = model.fit(self.inputs[train], self.targets[train], batch_size=16,
                                   epochs=100, validation_data=(self.x_testcnn, self.y_test))

            # Generate generalization metrics
            scores = model.evaluate(self.inputs[test], self.targets[test], verbose=0)
            print(
                f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
            self.acc_per_fold.append(scores[1] * 100)
            self.loss_per_fold.append(scores[0])

            # Increase fold number
            fold_no = fold_no + 1
        # == Provide average scores ==
        print('------------------------------------------------------------------------')
        print('Score per fold')
        for i in range(0, len(self.acc_per_fold)):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i + 1} - Loss: {self.loss_per_fold[i]} - Accuracy: {self.acc_per_fold[i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(self.acc_per_fold)} (+- {np.std(self.acc_per_fold)})')
        print(f'> Loss: {np.mean(self.loss_per_fold)}')
        print('------------------------------------------------------------------------')
        self.model_cnn = model

    def model_los(self):
        plt.plot(self.cnnhistory.history['loss'])
        plt.plot(self.cnnhistory.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def model_acc(self):
        plt.plot(self.cnnhistory.history['acc'])
        plt.plot(self.cnnhistory.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
