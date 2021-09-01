import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy.interpolate import interp1d


# loading classifier output data from a directory
def load_predictions(data_dir, prediction_dir, extra_signal=True):
    #y_test = np.load(os.path.join(data_dir, 'y_test.npy')) ## redundant
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    preds_matris = np.load(os.path.join(prediction_dir, 'preds_matris.npy'))
    if X_test.shape[0]==preds_matris.shape[-1]:
        pass
    ## check if predictions are only done on data
    elif X_test[X_test[:,-2]==1].shape[0]==preds_matris.shape[-1]:
        X_test = X_test[X_test[:,-2]==1]
    else:
        raise RuntimeError("Data and prediction shapes don't match!")
    val_loss_matris = np.load(os.path.join(prediction_dir, 'val_loss_matris.npy'))
    y_test = X_test[:,-2]

    if extra_signal:
        X_extrasig = np.load(os.path.join(data_dir, 'X_extrasig.npy'))
        preds_extrasig = np.load(os.path.join(prediction_dir, 'preds_matris_extrasig.npy'))

        X_test_combined = np.concatenate((X_test, X_extrasig), axis=0)
        y_test_combined = np.concatenate((y_test, np.ones(X_extrasig.shape[0])))
        preds_combined = np.concatenate((preds_matris, preds_extrasig), axis=-1)
    else:
        X_test_combined = X_test
        y_test_combined = y_test
        preds_combined = preds_matris

    return X_test_combined, y_test_combined, preds_combined, val_loss_matris


# returns list of minimum validation loss model paths
def minimum_validation_loss_models(prediction_dir, n_epochs=10):
    validation_loss_matrix = np.load(os.path.join(prediction_dir, 'val_loss_matris.npy'))
    model_paths = []
    for i in range(validation_loss_matrix.shape[0]):
        min_val_loss_epochs = np.argpartition(validation_loss_matrix[i, :],
                                              n_epochs)[:n_epochs]
        model_paths.append([os.path.join(prediction_dir, f"model_run{i}_ep{x}") for x in min_val_loss_epochs])
    return model_paths 


# load Tensorflow Models to derive predictions only on designated epochs
#   model_path_list should be a list of lists of model paths. The outer layer being per run,
#   the inner being per epoch: [[path_run0_ep0, path_run_ep1, ...], [path_run1_ep0, path_run1_ep1, ...], ...]
def preds_from_models(model_path_list, data_dir, save_dir, use_mjj=False, predict_on_samples=False, extra_signal=True, take_mean=True):

    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    if use_mjj:
        X_test = X_test.copy()
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        mjj_mean = np.mean(X_train[:, 0])
        mjj_std = np.std(X_train[:, 0])
        X_test[:, 0] -= mjj_mean
        X_test[:, 0] /= mjj_std
    if extra_signal:
        extrasig_data = np.load(os.path.join(data_dir, 'X_extrasig.npy'))
        if use_mjj:
            extrasig_data = extrasig_data.copy()
            extrasig_data[:, 0] -= mjj_mean
            extrasig_data[:, 0] /= mjj_std
    if predict_on_samples:
        test_data = X_test
    else:
        test_data = X_test[X_test[:,-2]==1]

    run_predictions = []
    run_predictions_extrasig = []
    for i, model_paths in enumerate(model_path_list): ## looping over runs
        model_list = [tf.keras.models.load_model(model_path) for model_path in model_paths]
        epoch_predictions = []
        epoch_predictions_extrasig = []
        for model in model_list: ## looping over epochs
            if use_mjj:
                epoch_predictions.append(model.predict(test_data[:,:5]).flatten())
                if extra_signal:
                    epoch_predictions_extrasig.append(model.predict(extrasig_data[:,:5]).flatten())
            else:
                epoch_predictions.append(model.predict(test_data[:,1:5]).flatten())
                if extra_signal:
                    epoch_predictions_extrasig.append(model.predict(extrasig_data[:,1:5]).flatten())
        run_predictions.append(np.stack(epoch_predictions))
        if extra_signal:
            run_predictions_extrasig.append(np.stack(epoch_predictions_extrasig))
   
    preds_matrix = np.stack(run_predictions)
    if take_mean:
        preds_matrix = np.mean(preds_matrix, axis=1, keepdims=True)
    np.save(os.path.join(save_dir, "preds_matris.npy"), preds_matrix)
    if extra_signal:
        preds_matrix_extrasig = np.stack(run_predictions_extrasig)
        if take_mean:
            preds_matrix_extrasig = np.mean(preds_matrix_extrasig, axis=1, keepdims=True)

        np.save(os.path.join(save_dir, "preds_matris_extrasig.npy"), preds_matrix_extrasig)
        return preds_matrix, preds_matrix_extrasig
    else:
        return preds_matrix


# combining the two functions above into one
def minimum_val_loss_model_evaluation(data_dir, prediction_dir, n_epochs=10, use_mjj=False,
                                    predict_on_samples=False, extra_signal=True, take_mean=True):
    model_paths = minimum_validation_loss_models(prediction_dir, n_epochs=n_epochs)
    return preds_from_models(model_paths, data_dir, prediction_dir, use_mjj=use_mjj,
      predict_on_samples=predict_on_samples, extra_signal=extra_signal, take_mean=take_mean)


# ensembling n epochs based on the lowest validation losses
def minumum_validation_loss_ensemble(predictions_matrix, validation_loss_matrix, n_epochs=10):
    min_val_loss_epochs = np.zeros((validation_loss_matrix.shape[0], n_epochs), dtype=int)
    for i in range(validation_loss_matrix.shape[0]):
        min_val_loss_epochs[i, :] = np.argpartition(validation_loss_matrix[i, :], n_epochs)\
            [:n_epochs]

    min_val_loss_preds = np.zeros((predictions_matrix.shape[0], predictions_matrix.shape[-1]))
    for i in range(predictions_matrix.shape[0]):
        min_val_loss_preds[i, :] = np.mean(predictions_matrix[i, min_val_loss_epochs[i, :], :],
                                           axis=0)
    min_val_loss_preds = min_val_loss_preds.reshape(min_val_loss_preds.shape[0], 1,
                                                    min_val_loss_preds.shape[1])

    return min_val_loss_preds


# evalautes the ROCs and SICs of one given training with minimum validation loss epoch picking
def full_single_evaluation(data_dir, prediction_dir, n_ensemble_epochs=10, extra_signal=True,
                           sic_range=(0,20), savefig=None, suppress_show=False, return_all=False):
    X_test, y_test, predictions, val_losses = load_predictions(
        data_dir, prediction_dir, extra_signal=extra_signal)
    if predictions.shape[1]==1: ## check if ensembling done already
        min_val_loss_predictions = predictions
    else:
        min_val_loss_predictions = minumum_validation_loss_ensemble(
            predictions, val_losses, n_epochs=n_ensemble_epochs)
    tprs, fprs, sics = tprs_fprs_sics(min_val_loss_predictions, y_test, X_test)

    return compare_on_various_runs(
        [tprs], [fprs], [np.zeros(min_val_loss_predictions.shape[0])], [""],
        sic_lim=sic_range, savefig=savefig, only_median=False, continuous_colors=False,
        reduced_legend=False, suppress_show=suppress_show, return_all=return_all)


# general curve plotting function for comparing collections of individual runs and their median
## TODO add some more documentation here
def compare_on_various_runs(tprs_list, fprs_list, pick_epochs_list, labels_list, sic_lim=None, savefig=None, only_median=False, continuous_colors=False, reduced_legend=False, suppress_show=False, return_all=False):
    assert len(tprs_list) == len(fprs_list) == len(pick_epochs_list) == len(labels_list), (
        "the input lists need to be of the same length")

    picked_median_colors = ["navy", "darkred", "darkgreen", "darkorange"]
    picked_single_colors = ["skyblue", "salmon", "lightgreen", "navajowhite"]
    if not continuous_colors and len(tprs_list)>len(picked_median_colors):
        print("for a non continuous color palette, additional colors need to be added "+\
              "to incorporate that many run collections")
        raise NotImplementedError
    if continuous_colors and not only_median:
        print("currently only support continuous colors on only_median runs")
        raise NotImplementedError

    # interpolation
    tprs_manual_list = []
    roc_median_list = []
    sic_median_list = []
    for run_collection in zip(tprs_list, fprs_list, pick_epochs_list):
        tprs, fprs, pick_epoch = run_collection
        max_min_tpr = 0.
        min_max_tpr = 1.
        for tpr in tprs.values():
            if min(tpr) > max_min_tpr:
                max_min_tpr = min(tpr)
            if max(tpr) < min_max_tpr:
                min_max_tpr = max(tpr)
        tprs_manual = np.linspace(max_min_tpr, min_max_tpr, 1000)
        tprs_manual_list.append(tprs_manual)

        roc_interpol = []
        sic_interpol = []
        for j in range(len(pick_epoch)):
            roc_function = interp1d(tprs[j, pick_epoch[j]], 1/fprs[j, pick_epoch[j]])
            sic_function = interp1d(tprs[j, pick_epoch[j]],
                                    tprs[j,pick_epoch[j]]/(fprs[j, pick_epoch[j]])**(0.5))
            roc_interpol.append(roc_function(tprs_manual))
            sic_interpol.append(sic_function(tprs_manual))
        roc_median_list.append(np.median(np.stack(roc_interpol), axis=0))
        sic_median_list.append(np.median(np.stack(sic_interpol), axis=0))

    # color map
    median_colors = cm.viridis(np.linspace(0., 0.95, len(tprs_list))) if continuous_colors \
        else picked_median_colors[:len(tprs_list)]
    if not only_median:
        single_colors = picked_single_colors[:len(tprs_list)]
    zorder_single = np.arange(0, 5*len(tprs_list), 5)
    zorder_median = np.arange(5*len(tprs_list)+5, 10*len(tprs_list)+5, 5)

    # draw ROCs
    plt.subplot(1, 2, 1)
    for k, run_collection in enumerate(zip(tprs_list, fprs_list, pick_epochs_list, labels_list)):
        tprs, fprs, pick_epoch, label = run_collection
        full_label = "" if label=="" else ", "+label
        if not only_median:
            #for j in range(len(pick_epoch)):
            for j, picked_epoch in enumerate(pick_epoch):
                #plt.plot(tprs[j,pick_epoch[j]], 1/fprs[j,pick_epoch[j]], color=single_colors[k])
                plt.plot(tprs[j, picked_epoch], 1/fprs[j, picked_epoch], color=single_colors[k])
            if not reduced_legend:
                plt.plot(np.nan, np.nan, color=single_colors[k],
                         label=f"{len(pick_epoch)} individual runs{full_label}",
                         zorder=zorder_single[k])
        plt.plot(tprs_manual_list[k], roc_median_list[k], color=median_colors[k],
                 label=f"median{full_label}", zorder=zorder_median[k])
    plt.plot(np.linspace(0.0001, 1, 300), 1/np.linspace(0.0001, 1, 300),
             color="gray", linestyle=":", label="random")
    plt.title("Signal Region", loc="right", style='italic')
    plt.xlabel('Signal Efficiency (True Positive Rate)')
    plt.ylabel('Rejection (1/False Positive Rate)')
    plt.legend(loc='upper right')
    plt.yscale('log')

    # draw SICs
    plt.subplot(1, 2, 2)
    for k, run_collection in enumerate(zip(tprs_list, fprs_list, pick_epochs_list, labels_list)):
        tprs, fprs, pick_epoch, label = run_collection
        full_label = "" if label=="" else ", "+label
        if not only_median:
            #for j in range(len(pick_epoch)):
            for j, picked_epoch in enumerate(pick_epoch):
                #plt.plot(tprs[j,pick_epoch[j]],
                #tprs[j,pick_epoch[j]]/(fprs[j,pick_epoch[j]])**(0.5), color=single_colors[k])
                plt.plot(tprs[j, picked_epoch],
                         tprs[j, picked_epoch]/(fprs[j, picked_epoch])**(0.5),
                         color=single_colors[k])
            if not reduced_legend:
                plt.plot(np.nan, np.nan, color=single_colors[k],
                         label=f"{len(pick_epoch)} individual runs{full_label}",
                         zorder=zorder_single[k])
        plt.plot(tprs_manual_list[k], sic_median_list[k], color=median_colors[k],
                 label=f"median{full_label}", zorder=zorder_median[k])
    plt.plot(np.linspace(0.0001, 1, 300),
             np.linspace(0.0001, 1, 300)/np.linspace(0.0001, 1, 300)**(0.5), color="gray",
             linestyle=":", label="random")
    plt.ylim(sic_lim)
    plt.title("Signal Region", loc="right", style='italic')
    plt.ylabel('Significance Improvement')
    plt.xlabel('Signal Efficiency (True Positive Rate)')
    plt.legend(loc='upper right')

    # save / display
    plt.subplots_adjust(right=2.0)
    #plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, bbox_inches="tight")
    if not suppress_show:
        plt.show()

    if return_all:
        return tprs_manual_list, roc_median_list, sic_median_list, tprs_list, fprs_list
    else:
        return tprs_manual_list, roc_median_list, sic_median_list


# tprs and tprs extraction function for the signal vs background task
def tprs_fprs_sics(preds_matris, y_test, X_test):
    runs = preds_matris.shape[0]
    epchs = preds_matris.shape[1]
    tprs = {}
    fprs = {}
    sics = {}
    for j in range(runs):
        for i in range(epchs):
            fpr, tpr, thresholds = roc_curve(X_test[:, -1][y_test == 1],
                                             preds_matris[j, i][y_test == 1])
            fpr_nonzero = np.delete(fpr, np.argwhere(fpr == 0))
            tpr_nonzero = np.delete(tpr, np.argwhere(fpr == 0))
            tprs[j, i] = tpr_nonzero
            fprs[j, i] = fpr_nonzero
            sics[j, i] = tprs[j, i]/fprs[j, i]**0.5
    return tprs, fprs, sics


# tprs and tprs extraction function for the data vs sample task
def tprs_fprs_sics_data_vs_sample(preds_matris, y_test):
    runs = preds_matris.shape[0]
    epchs = preds_matris.shape[1]
    tprs = {}
    fprs = {}
    sics = {}
    for j in range(runs):
        for i in range(epchs):
            fpr, tpr, thresholds = roc_curve(y_test, preds_matris[j, i])
            fpr_nonzero = np.delete(fpr, np.argwhere(fpr == 0))
            tpr_nonzero = np.delete(tpr, np.argwhere(fpr == 0))
            tprs[j, i] = tpr_nonzero
            fprs[j, i] = fpr_nonzero
            sics[j, i] = tprs[j, i]/fprs[j, i]**0.5
    return tprs, fprs, sics


# ensemble preds function -- TODO variable number of epochs
def ensemble_preds_func(preds_matris, y_test):
    bce = tf.keras.losses.BinaryCrossentropy()
    ensemble_val_losses = []
    ensemble_preds_list = []
    for i in range(preds_matris.shape[0]):
        ensemble_preds = 0.1*(preds_matris[i, :-9]
                              +preds_matris[i, 1:-8]
                              +preds_matris[i, 2:-7]
                              +preds_matris[i, 3:-6]
                              +preds_matris[i, 4:-5]
                              +preds_matris[i, 5:-4]
                              +preds_matris[i, 6:-3]
                              +preds_matris[i, 7:-2]
                              +preds_matris[i, 8:-1]
                              +preds_matris[i, 9:])
        ensemble_val_loss = []
        for j in range(ensemble_preds.shape[0]):
            ensemble_val_loss.append(bce(y_test, ensemble_preds[j, :]).numpy().item())
        ensemble_val_losses.append(np.array(ensemble_val_loss))
        ensemble_preds_list.append(ensemble_preds)
    ensemble_val_loss_matris = np.stack(ensemble_val_losses)
    ensemble_preds_matris = np.stack(ensemble_preds_list)
    return ensemble_preds_matris, ensemble_val_loss_matris


# Find the minimal val_loss for each training, and which epoch it belongs to
def min_val_loss_epoch(val_loss_matris):
    min_val_loss = []
    epoch_min_val_loss = []
    for j in range(val_loss_matris.shape[0]):
        min_val_loss.append(np.min(val_loss_matris[j, :]))
        epoch_min_val_loss.append(np.argmin(val_loss_matris[j, :]))
    return epoch_min_val_loss


# Quick evaluation of classic ANODE
def classic_ANODE_eval(predsdir, savefig=None, suppress_show=False, extra_signal=True, return_all=False):
    preds = np.load(os.path.join(predsdir, "preds_matris.npy"))
    labels = np.load(os.path.join(predsdir, "sig_labels.npy"))
    if extra_signal:
        preds_extrasig = np.load(os.path.join(predsdir, "preds_matris_extrasig.npy"))

    # removing nans and infs
    nan_mask = np.logical_and(~np.isnan(preds), ~np.isinf(preds))
    preds = preds[:,:,nan_mask[0,0,:]]
    labels = labels[nan_mask[0,0,:]]
    if extra_signal:
        nan_mask = np.logical_and(~np.isnan(preds_extrasig), ~np.isinf(preds_extrasig))
        preds_extrasig = preds_extrasig[:,:,nan_mask[0,0,:]]

    if extra_signal:
        preds_combined = np.concatenate((preds, preds_extrasig), axis=-1)
        labels_combined = np.concatenate((labels, np.ones(preds_extrasig.shape[-1])))
    else:
        preds_combined = preds
        labels_combined = labels

    tprs, fprs, sics = tprs_fprs_sics(preds_combined, np.ones(preds_combined.shape[-1]), labels_combined.reshape(-1,1))
    fprs_list = [fprs]
    tprs_list = [tprs]
    pick_epochs_list = [np.zeros(1)]
    labels_list = [""]

    return compare_on_various_runs(tprs_list, fprs_list, pick_epochs_list, labels_list, savefig=savefig, suppress_show=suppress_show, return_all=return_all)


# evaluate the performance of the classifier
def evaluate_performance(tprs, fprs, eval_range=(0.18, 0.23)):
    performance_results = []
    for key in tprs.keys():
        sics = tprs[key]/np.sqrt(fprs[key])
        # average sic over eval_range
        performance_results.append(sics[(min(eval_range) < tprs[key]) &
                                        (tprs[key] < max(eval_range))].mean())
    performance_results = np.array(performance_results)
    return performance_results.mean(), performance_results.std()


# calculate the average performance over the trainings of the classifier
def get_average_performance(data_dir, prediction_dir, n_ensemble_epochs=10,
                            extra_signal=True, eval_range=(0.18, 0.23)):
    X_test, y_test, predictions, val_losses = load_predictions(
        data_dir, prediction_dir, extra_signal=extra_signal)
    if predictions.shape[1] == 1:  # check if ensembling done already
        min_val_loss_predictions = predictions
    else:
        min_val_loss_predictions = minumum_validation_loss_ensemble(
            predictions, val_losses, n_epochs=n_ensemble_epochs)
    tprs, fprs, sics = tprs_fprs_sics(min_val_loss_predictions, y_test, X_test)
    return evaluate_performance(tprs, fprs, eval_range)
