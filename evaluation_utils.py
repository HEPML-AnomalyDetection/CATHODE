import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from scipy.special import logit


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
        print("minimum validation loss epochs:", min_val_loss_epochs)
        model_paths.append([os.path.join(prediction_dir, f"model_run{i}_ep{x}") for x in min_val_loss_epochs])
    return model_paths 


# load Pytorch Models to derive predictions only on designated epochs
#   model_path_list should be a list of lists of model paths. The outer layer being per run,
#   the inner being per epoch: [[path_run0_ep0, path_run_ep1, ...], [path_run1_ep0, path_run1_ep1, ...], ...]
def preds_from_models(model_path_list, data_dir, save_dir, use_mjj=False, predict_on_samples=False, extra_signal=True, take_mean=True):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu") 

    X_test = np.load(os.path.join(data_dir, 'X_test.npy')).astype("float32")
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
        model_list = [torch.load(model_path, map_location=device) for model_path in model_paths]
        for model in model_list:
            model.eval()
        epoch_predictions = []
        epoch_predictions_extrasig = []
        for model in model_list: ## looping over epochs
            if use_mjj:
                epoch_predictions.append(model.predict(test_data[:,:-2]).flatten())
                if extra_signal:
                    epoch_predictions_extrasig.append(model.predict(extrasig_data[:,:-2]).flatten())
            else:
                epoch_predictions.append(model.predict(test_data[:,1:-2]).flatten())
                if extra_signal:
                    epoch_predictions_extrasig.append(model.predict(extrasig_data[:,1:-2]).flatten())
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
    #bce = tf.keras.losses.BinaryCrossentropy()
    bce = torch.nn.BCELoss()
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
            #ensemble_val_loss.append(bce(y_test, ensemble_preds[j, :]).numpy().item())
            ensemble_val_loss.append(bce(ensemble_preds[j, :], y_test).numpy().item())
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


def plot_data_sample_comparison(X_vals, y_vals, nbins=50, alpha=0.5,
                                title=None, savefig=None,
                                data_label=None, sample_label=None,
                                data_color=None, sample_color=None,
                                step_hist=False, draw_signal=False,
                                remove_signal=True, signal_color=None,
                                force_ranges=None, logit_trsf=False):
    # Sanity check plot comparing data and samples before running
    # the classifier. In version it also comes with 2D correlation
    # contours. One can also provide a list of forced ranges to
    # track down problems in hindsight.

    if data_label is None:
        data_label = "Data"
    if sample_label is None:
        sample_label = "Samples"
    if data_color is None:
        data_color = "tab:blue"
    if sample_color is None:
        sample_color = "tab:orange"
    if signal_color is None:
        signal_color = "magenta"

    sample_hist_type = "step" if step_hist else "bar"

    if logit_trsf:
        X_features = X_vals[:, :-2]
        min_X = np.min(X_features, axis=0) - 1e-6
        max_X = np.max(X_features, axis=0) + 1e-5
        X_features = (X_features-min_X)/(max_X-min_X)
        X_features = logit(X_features)
        X_features = ((X_features - np.mean(X_features, axis=0))
                      / np.std(X_features, axis=0))
        X_vals = np.hstack((X_features, X_vals[:, -2:]))

    data_array = X_vals[y_vals == 1]
    samp_array = X_vals[y_vals == 0]
    signal_array = data_array[data_array[:, -1] == 1]
    if remove_signal:
        data_array = data_array[data_array[:, -1] == 0]

    n_features = data_array.shape[1]-2

    if force_ranges is not None:
        assert_msg = ("force_ranges needs to be list with the same length as "
                      + "number of features (cond+aux)!")
        assert len(force_ranges) == n_features, assert_msg
        assert_msg = ("elements in force_range need to be tuples describing "
                      + "the feature ranges in standardized representation "
                      + "or None if no range should be enforced!")
        for rng in force_ranges:
            assert rng is None or len(rng) == 2, assert_msg
    else:
        force_ranges = [None for x in range(n_features)]

    fig = plt.figure(figsize=(2.5*n_features, 2.5*n_features))
    gs = gridspec.GridSpec(n_features*2, n_features*2)

    # uniformizing plotting ranges
    feature_ranges = []
    for i in range(n_features):
        if force_ranges[i] is None:
            feature_range = (min(data_array[:, i]), max(data_array[:, i]))
        else:
            feature_range = force_ranges[i]
        bin_length = (feature_range[1]-feature_range[0])/nbins
        feature_ranges.append(
            (feature_range[0]-0.5*bin_length, feature_range[1]+0.5*bin_length))

    for i in range(n_features):
        grid_row_number = 2*i
        for j in range(n_features):
            grid_col_number = 2*j

            # separate legend
            if i == 0 and j == 1:
                plt.subplot(gs[grid_row_number:grid_row_number+2,
                            grid_col_number:grid_col_number+2])
                plt.plot(np.nan, np.nan, color=data_color,
                         label=data_label)
                plt.plot(np.nan, np.nan, color=sample_color,
                         label=sample_label)
                if draw_signal:
                    plt.plot(np.nan, np.nan, color=signal_color,
                             label="Signal")
                ax = plt.gca()
                ax.axis("off")
                plt.legend(loc="center", frameon=False)

            # only drawing half of symmetric correlation matrix
            if j > i:
                continue

            plt.subplot(gs[grid_row_number:grid_row_number+2,
                        grid_col_number:grid_col_number+2])

            if i == j:
                # marginal histograms
                binning = np.linspace(
                    feature_ranges[i][0], feature_ranges[i][1], nbins+1)

                plt.hist(
                    data_array[:, i], bins=binning, density=True,
                    color=data_color, alpha=alpha)
                if len(samp_array) != 0:
                    plt.hist(
                        samp_array[:, i], bins=binning, density=True,
                        color=sample_color, alpha=alpha,
                        histtype=sample_hist_type)
                if draw_signal:
                    plt.hist(
                        signal_array[:, i], bins=binning, density=True,
                        color=signal_color, alpha=alpha,
                        histtype=sample_hist_type)
                plt.xlim(feature_ranges[i])
                plt.plot(np.nan, np.nan, linestyle="none", label="(marginal)")
                plt.legend(loc="upper right", frameon=False)

            else:
                # 2D correlation contours
                ybinning = np.linspace(
                    feature_ranges[i][0], feature_ranges[i][1], nbins+1)
                xbinning = np.linspace(
                    feature_ranges[j][0], feature_ranges[j][1], nbins+1)

                data_counts, xbinning, ybinning = np.histogram2d(
                    data_array[:, j], data_array[:, i],
                    bins=(xbinning, ybinning), normed=True)
                extent = [*feature_ranges[j], *feature_ranges[i]]
                plt.contour(data_counts.transpose(), colors=[data_color],
                            extent=extent)
                if len(samp_array) != 0:
                    samp_counts, _, _ = np.histogram2d(
                        samp_array[:, j], samp_array[:, i],
                        bins=(xbinning, ybinning), normed=True)
                    plt.contour(samp_counts.transpose(),
                                colors=[sample_color], extent=extent)
                if draw_signal:
                    signal_counts, _, _ = np.histogram2d(
                        signal_array[:, j], signal_array[:, i],
                        bins=(xbinning, ybinning), normed=True)
                    plt.contour(signal_counts.transpose(),
                                colors=[signal_color], extent=extent)

            # only showing labels at the border
            if i == n_features-1:
                if j == 0:
                    plt.xlabel("cond feature")
                else:
                    plt.xlabel(f"aux feature {j}")
            else:
                ax = plt.gca()
                ax.axes.xaxis.set_ticklabels([])
            if j == 0:
                if i == 0:
                    plt.ylabel("Events (a.u.)")
                else:
                    plt.ylabel(f"aux feature {i}")
            else:
                ax = plt.gca()
                ax.axes.yaxis.set_ticklabels([])

    if title is not None:
        plt.suptitle(title)
    fig.tight_layout()

    if savefig is not None:
        plt.savefig(savefig+".pdf", bbox_inches="tight")
    else:
        plt.show()
