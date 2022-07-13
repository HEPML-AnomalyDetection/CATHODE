import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from classifier import build_classifier
from sklearn.utils import class_weight


def train_model(classifier_configfile, epochs, X_train, y_train, X_test, y_test, X_extrasig=None,
                X_val=None, use_mjj=False, batch_size=128, supervised=False,
                use_class_weights=False, CWoLa=False, SR_center=3.5, save_model=None, verbose=True):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
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
            input_val = X_val[:, :-2]
        else:
            input_val = X_test[:, :-2]
        input_train = X_train[:, :-2]

    else:
        input_train = X_train[:, 1:-2]
        if X_val is not None:
            input_val = X_val[:, 1:-2]
        else:
            input_val = X_test[:, 1:-2]

    if supervised:
        print("Running a fully supervised training. Sig/bkg labels will be known!")
        input_train = input_train[y_train == 1]
        input_val = input_val[X_val[:, -2] == 1]
        label_train = X_train[y_train == 1][:, -1]
        label_val = X_val[X_val[:, -2] == 1][:, -1]
    else:
        label_train = y_train
        if X_val is not None:
            label_val = X_val[:, -2]
        else:
            label_val = y_test

    if use_class_weights:
        if CWoLa:
            # training weights
            lower_SB_mask_train = np.logical_and((y_train == 0), (X_train[:, 0] < SR_center))
            upper_SB_mask_train = np.logical_and((y_train == 0), (X_train[:, 0] > SR_center))
            SB_labels_train = np.concatenate((np.zeros(sum(lower_SB_mask_train)),
                                        np.ones(sum(upper_SB_mask_train)))).astype('int64')
            ## weights between left and right SB
            SB_weights_train  = len(SB_labels_train) / (2 * np.bincount(SB_labels_train))
            SB_vs_SR_events_train = np.array([len(SB_labels_train), sum(y_train == 1)])
            SB_vs_SR_weights_train = len(y_train) / (2 * SB_vs_SR_events_train) ## weights between SB and SR

            SR_weight_train = SB_vs_SR_weights_train[1]
            lower_SB_weight_train = SB_vs_SR_weights_train[0]*SB_weights_train[0]
            upper_SB_weight_train = SB_vs_SR_weights_train[0]*SB_weights_train[1]

            sample_weights_train = lower_SB_weight_train*lower_SB_mask_train + SR_weight_train*y_train +\
                upper_SB_weight_train*upper_SB_mask_train

            # validation weights
            lower_SB_mask_val = np.logical_and((label_val == 0), (input_val[:, 0] < SR_center))
            upper_SB_mask_val = np.logical_and((label_val == 0), (input_val[:, 0] > SR_center))
            SB_labels_val = np.concatenate((np.zeros(sum(lower_SB_mask_val)),
                                        np.ones(sum(upper_SB_mask_val)))).astype('int64')
            ## weights between left and right SB
            SB_weights_val  = len(SB_labels_val) / (2 * np.bincount(SB_labels_val))
            SB_vs_SR_events_val = np.array([len(SB_labels_val), sum(label_val == 1)])
            SB_vs_SR_weights_val = len(label_val) / (2 * SB_vs_SR_events_val) ## weights between SB and SR

            SR_weight_val = SB_vs_SR_weights_val[1]
            lower_SB_weight_val = SB_vs_SR_weights_val[0]*SB_weights_val[0]
            upper_SB_weight_val = SB_vs_SR_weights_val[0]*SB_weights_val[1]

            sample_weights_val = lower_SB_weight_val*lower_SB_mask_val + SR_weight_val*label_val +\
                upper_SB_weight_val*upper_SB_mask_val

            sample_weights = (sample_weights_train, sample_weights_val)
            class_weights = None
        else:
            class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                              classes=np.unique(label_train),
                                                              y=label_train)
            class_weights = dict(enumerate(class_weights))
            sample_weights = None
    else:
        class_weights = None
        sample_weights = None

    if sample_weights is not None:
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(input_train),
            torch.tensor(label_train).reshape(-1,1),
            torch.tensor(sample_weights[0]).reshape(-1,1))
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(input_val),
            torch.tensor(label_val).reshape(-1,1),
            torch.tensor(sample_weights[1]).reshape(-1,1))
    else:
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(input_train),
            torch.tensor(label_train).reshape(-1,1))
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(input_val),
            torch.tensor(label_val).reshape(-1,1))

    train_dataloader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
                      val_dataset, batch_size=batch_size, shuffle=False)

    train_loss = np.zeros(epochs) ## add pre-training loss?
    val_loss = np.zeros(epochs)

    model, loss_func, optimizer = build_classifier(classifier_configfile, n_inputs=input_train.shape[1])
    model.to(device)
    assert not (sample_weights is not None and class_weights is not None), (
        "Both sample weights and class weights given!")
    for epoch in range(epochs):
        print("training epoch nr", epoch)
        epoch_train_loss = 0.
        epoch_val_loss = 0.

        model.train()
        for i, batch in enumerate(train_dataloader):
            if verbose:
                print("...batch nr", i)
            if sample_weights is not None:
                batch_inputs, batch_labels, batch_weights = batch
                batch_weights = batch_weights.to(device)
                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            else:
                batch_inputs, batch_labels = batch
                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                if class_weights is not None:
                    batch_weights = (torch.ones(batch_labels.shape, device=device)
                        - batch_labels)*class_weights[0] \
                        + batch_labels*class_weights[1]
                else:
                    batch_weights = None

            optimizer.zero_grad()
            batch_outputs = model(batch_inputs)
            batch_loss = loss_func(batch_outputs, batch_labels, weight=batch_weights)
            batch_loss.backward()
            optimizer.step()
            epoch_train_loss += batch_loss
            if verbose:
                print("...batch training loss:", batch_loss.item())

        epoch_train_loss /= (i+1)
        print("training loss:", epoch_train_loss.item())

        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(val_dataloader):

                if sample_weights is not None:
                    batch_inputs, batch_labels, batch_weights = batch
                    batch_weights = batch_weights.to(device)
                    batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                else:
                    batch_inputs, batch_labels = batch
                    batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                    if class_weights is not None:
                        batch_weights = (torch.ones(batch_labels.shape, device=device)
                            - batch_labels)*class_weights[0] \
                            + batch_labels*class_weights[1]
                    else:
                        batch_weights = None

                batch_outputs = model(batch_inputs)
                batch_loss = loss_func(batch_outputs, batch_labels, weight=batch_weights)
                epoch_val_loss += batch_loss
            epoch_val_loss /= (i+1)
        print("validation loss:", epoch_val_loss.item())

        train_loss[epoch] = epoch_train_loss
        val_loss[epoch] = epoch_val_loss

        if save_model is not None:
            torch.save(model, save_model+"_ep"+str(epoch))
    return train_loss, val_loss


def train_n_models(n_runs, classifier_configfile, epochs, X_train, y_train, X_test, y_test,
                   X_extrasig=None, X_val=None, use_mjj=False, batch_size=128, supervised=False,
                   use_class_weights=False, CWoLa=False, SR_center=3.5, verbose=True,
                   savedir=None, save_model=None):
    # Trains n models and records the resulting losses and test data predictions.
    #    The outputs are saved to files if a directory path is given to savedir.
    #    If supervised is set true, the classifier learns to distinguish sig and
    #    bkg according to their actual labels.

    loss = {}
    val_loss = {}

    for j in range(n_runs):
        print(f"Training model nr {j}...")
        if save_model is not None:
            current_save_model = save_model+"_run"+str(j)
        else:
            current_save_model = None
        loss[j], val_loss[j] = train_model(
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
        raise NotImplementedError("Removed prediction saving.",
            "Please provide model name to save_model.")
    else:
        preds_matrix = None
        preds_matrix_extrasig = None

    return loss_matrix, val_loss_matrix


def plot_classifier_losses(train_losses, val_losses, yrange=None, savefig=None, suppress_show=False):
    # plots the classifier training losses from loss array. The image is saved if a filename is
    # given to the savefig parameter
    avg_train_losses = (
        train_losses[5:]+train_losses[4:-1]+train_losses[3:-2]
        +train_losses[2:-3]+train_losses[1:-4])/5
    avg_val_losses = (val_losses[5:]+val_losses[4:-1]+val_losses[3:-2]
                      +val_losses[2:-3]+val_losses[1:-4])/5

    plt.plot(range(1, len(train_losses)), train_losses[1:], linestyle=":", color="blue")
    plt.plot(range(1, len(val_losses)), val_losses[1:], linestyle=":", color="orange")
    plt.plot(range(3, len(train_losses)-2), avg_train_losses, label="Training", color="blue")
    plt.plot(range(3, len(val_losses)-2), avg_val_losses, label="Validation", color="orange")
    plt.plot(np.nan, np.nan, linestyle="None", label=" ")
    plt.plot(np.nan, np.nan, linestyle=":", color="black", label="Per Epoch Value")
    plt.plot(np.nan, np.nan, linestyle="-", color="black", label="5-Epoch Average")

    if yrange is not None:
        plt.ylim(*yrange)
    plt.xlabel("Training Epoch")
    plt.ylabel("(Mean) Binary Cross Entropy Loss")
    plt.legend(loc="upper right", frameon=False)
    if savefig is not None:
        plt.savefig(savefig+".pdf", bbox_inches="tight")
    if not suppress_show:
        plt.show()
    plt.close()
