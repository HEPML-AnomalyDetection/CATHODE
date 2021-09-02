import os
import numpy as np
from keras.callbacks import Callback
from classifier import build_classifier
from sklearn.utils import class_weight


class prediction_history(Callback):
    # custom callback that stores each epoch prediction
    def __init__(self, model, X_train, X_test, X_extrasig=None, use_mjj=False, save_model=None):
        self.predhis = []
        self.predhis_extrasig = []
        self.use_mjj = use_mjj
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.X_extrasig = X_extrasig
        self.save_model = save_model

    def on_epoch_end(self, epoch, logs={}):
        if self.save_model is None:
            if self.use_mjj:
                self.predhis.append(self.model.predict(self.X_test[:, :5]))
            else:
                self.predhis.append(self.model.predict(self.X_test[:, 1:5]))
            if self.X_extrasig is not None:
                if self.use_mjj:
                    self.predhis_extrasig.append(self.model.predict(self.X_extrasig[:, :5]))
                else:
                    self.predhis_extrasig.append(self.model.predict(self.X_extrasig[:, 1:5]))
        else:
            self.model.save(self.save_model+"_ep"+str(epoch))


def train_model(classifier_configfile, epochs, X_train, y_train, X_test, y_test, X_extrasig=None,
                X_val=None, use_mjj=False, batch_size=128, supervised=False,
                use_class_weights=False, CWoLa=False, SR_center=3.5, save_model=None, verbose=True):
    # training a single classifier model
    if supervised:
        assert X_val is not None, (
            "Validation data needs to be provided in order to run supervised training!")

    if use_mjj:
        X_train = X_train.copy()
        X_test = X_test.copy()
        mjj_mean = np.mean(X_train[:, 0])
        mjj_std = np.std(X_train[:, 0])
        X_train[:, 0] -= mjj_mean
        X_train[:, 0] /= mjj_std
        X_test[:, 0] -= mjj_mean
        X_test[:, 0] /= mjj_std
        if X_extrasig is not None:
            X_extrasig = X_extrasig.copy()
            X_extrasig[:, 0] -= mjj_mean
            X_extrasig[:, 0] /= mjj_std
        if X_val is not None:
            X_val = X_val.copy()
            X_val[:, 0] -= mjj_mean
            X_val[:, 0] /= mjj_std
            input_val = X_val[:, :5]
        else:
            input_val = X_test[:, :5]
        input_train = X_train[:, :5]

    else:
        input_train = X_train[:, 1:5]
        if X_val is not None:
            input_val = X_val[:, 1:5]
        else:
            input_val = X_test[:, 1:5]

    if supervised:
        print("Running a fully supervised training. Sig/bkg labels will be known!")
        input_train = input_train[y_train == 1]
        input_val = input_val[X_val[:, -2] == 1]
        label_train = X_train[y_train == 1][:, 6]
        label_val = X_val[X_val[:, -2] == 1][:, 6]
    else:
        label_train = y_train
        if X_val is not None:
            label_val = X_val[:, -2]
        else:
            label_val = y_test

    if use_class_weights:
        if CWoLa:
            lower_SB_mask = np.logical_and((y_train == 0), (X_train[:, 0] < SR_center))
            upper_SB_mask = np.logical_and((y_train == 0), (X_train[:, 0] > SR_center))
            SB_labels = np.concatenate((np.zeros(sum(lower_SB_mask)),
                                        np.ones(sum(upper_SB_mask)))).astype('int64')
            # weights between left and right SB
            SB_weights = len(SB_labels) / (2 * np.bincount(SB_labels))
            SB_vs_SR_events = np.array([len(SB_labels), sum(y_train == 1)])
            SB_vs_SR_weights = len(y_train) / (2 * SB_vs_SR_events) # weights between SB and SR

            SR_weight = SB_vs_SR_weights[1]
            lower_SB_weight = SB_vs_SR_weights[0]*SB_weights[0]
            upper_SB_weight = SB_vs_SR_weights[0]*SB_weights[1]

            sample_weights = lower_SB_weight*lower_SB_mask + SR_weight*y_train +\
                upper_SB_weight*upper_SB_mask
            class_weights = None
        else:
            class_weights = class_weight.compute_class_weight('balanced',
                                                              np.unique(label_train),
                                                              label_train)
            class_weights = dict(enumerate(class_weights))
            sample_weights = None
    else:
        class_weights = None
        sample_weights = None

    model = build_classifier(classifier_configfile)
    predictions = prediction_history(model, X_train, X_test, X_extrasig=X_extrasig,
                                     use_mjj=use_mjj, save_model=save_model)
    kerasfit = model.fit(input_train, label_train, epochs=epochs, batch_size=batch_size,
                         verbose=verbose, class_weight=class_weights, sample_weight=sample_weights,
                         validation_data=(input_val, label_val), callbacks=[predictions])

    return predictions.predhis, predictions.predhis_extrasig, kerasfit.history["loss"], kerasfit.history["val_loss"]


def train_n_models(n_runs, classifier_configfile, epochs, X_train, y_train, X_test, y_test,
                   X_extrasig=None, X_val=None, use_mjj=False, batch_size=128, supervised=False,
                   use_class_weights=False, CWoLa=False, SR_center=3.5, verbose=True,
                   savedir=None, save_model=None):
    # Trains n models and records the resulting losses and test data predictions.
    #    The outputs are saved to files if a directory path is given to savedir.
    #    If supervised is set true, the classifier learns to distinguish sig and
    #    bkg according to their actual labels.

    preds = {}
    preds_extrasig = {}
    loss = {}
    val_loss = {}

    for j in range(n_runs):
        print(f"Training model nr {j}...")
        if save_model is not None:
            current_save_model = save_model+"_run"+str(j)
        else:
            current_save_model = None
        preds[j], preds_extrasig[j], loss[j], val_loss[j] = train_model(
            classifier_configfile, epochs, X_train, y_train, X_test, y_test,
            X_val=X_val, X_extrasig=X_extrasig, use_mjj=use_mjj, batch_size=batch_size,
            supervised=supervised, use_class_weights=use_class_weights,
            CWoLa=CWoLa, SR_center=SR_center, save_model=current_save_model,
            verbose=verbose)

    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)

    val_loss_matrix = np.zeros((n_runs, epochs))
    for j in range(n_runs):
        for i in range(epochs):
            val_loss_matrix[j, i] = val_loss[j][i]
    loss_matrix = np.zeros((n_runs, epochs))
    for j in range(n_runs):
        for i in range(epochs):
            loss_matrix[j, i] = loss[j][i]

    np.save(os.path.join(savedir, 'val_loss_matris.npy'), val_loss_matrix)
    np.save(os.path.join(savedir, 'loss_matris.npy'), loss_matrix)

    if save_model is None:
        preds_matrix = np.zeros((n_runs, epochs, len(preds[0][0])))
        for j in range(n_runs):
            for i in range(epochs):
                for k in range(len(preds[0][0])):
                    preds_matrix[j, i, k] = preds[j][i][k]

        if X_extrasig is None:
            preds_matrix_extrasig = None
        else:
            preds_matrix_extrasig = np.zeros((n_runs, epochs, len(preds_extrasig[0][0])))
            for j in range(n_runs):
                for i in range(epochs):
                    for k in range(len(preds_extrasig[0][0])):
                        preds_matrix_extrasig[j, i, k] = preds_extrasig[j][i][k]

        np.save(os.path.join(savedir, 'preds_matris.npy'), preds_matrix)
        if X_extrasig is not None:
            np.save(os.path.join(savedir, 'preds_matris_extrasig.npy'), preds_matrix_extrasig)
    else:
        preds_matrix = None
        preds_matrix_extrasig = None

    return preds_matrix, preds_matrix_extrasig, loss_matrix, val_loss_matrix
