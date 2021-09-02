import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from tqdm import tqdm
import flows as fnn


def compute_likelihoods(model_list, data_loader, minval, maxval, meanval, stdval, device, verbose=False):
    # compute the likelihood for each point in the transformed data loader according to a list of loaded models
    for model in model_list:
        model.eval()

    full_loss_list = []

    with torch.no_grad():

        # create a progress bar using tqdm
        if verbose:
            pbar = tqdm(total=len(data_loader.dataset))

        for batch_idx, batch_data in enumerate(data_loader):
            data = batch_data[0]
            data = data.to(device)
            cond_data = batch_data[1].float()
            cond_data = cond_data.to(device)

            loss_vals_list = []
            for model in model_list:
                loss_vals_raw = model.log_probs(data, cond_data).flatten()
                loss_vals_list.append(loss_vals_raw + torch.sum(torch.log(2 * (1 + torch.cosh(data * stdval + meanval)) / (stdval * (maxval - minval))), dim=1))
            loss_vals = torch.stack(loss_vals_list)

            if len(model_list) > 1: 
                loss_vals = torch.exp(loss_vals) # averaging in exp space
                loss_vals = torch.mean(loss_vals, dim=0)
                loss_vals = torch.log(loss_vals).flatten() # back to log space
            else:
                loss_vals = loss_vals.flatten()
            full_loss_list.append(loss_vals.detach())

            if verbose:
                pbar.update(data.shape[0])

    return torch.cat(full_loss_list)


def compute_ANODE_score(inner_model_list, outer_model_list, data_handler,
                        device, extra_signal=True, savedir=None):
    # compute the anomaly score from ANODE on a given data_handler object.
    #   The methods data_handler.data.preprocess_ANODE_data() and 
    #   data_handler.preprocess_classic_ANODE_data() should be called first.
    #   The predictions are saved when a corresponding directory is provided as savedir.
    ## TODO add extra_signal conditional to suppress the extra signal prediction

    assert data_handler.inner_classic_ANODE_outermodel_datadict_train is not None, "The method data_handler.preprocess_classic_ANODE_data() needs to be called first!"

    minval = data_handler.inner_classic_ANODE_outermodel_datadict_train['min']
    maxval = data_handler.inner_classic_ANODE_outermodel_datadict_train['max']
    meanval = data_handler.inner_classic_ANODE_outermodel_datadict_train['mean2']
    stdval = data_handler.inner_classic_ANODE_outermodel_datadict_train['std2']

    log_p_bkg_test = compute_likelihoods(outer_model_list, data_handler.inner_classic_ANODE_outermodel_datadict_test['loader'], minval, maxval, meanval, stdval, device, verbose=False)
    if extra_signal:
        log_p_bkg_extrasig = compute_likelihoods(outer_model_list, data_handler.inner_classic_ANODE_outermodel_datadict_extrasig['loader'], minval, maxval, meanval, stdval, device, verbose=False)

    minval = data_handler.inner_classic_ANODE_innermodel_datadict_train['min']
    maxval = data_handler.inner_classic_ANODE_innermodel_datadict_train['max']
    meanval = data_handler.inner_classic_ANODE_innermodel_datadict_train['mean2']
    stdval = data_handler.inner_classic_ANODE_innermodel_datadict_train['std2']

    log_p_data_test = compute_likelihoods(inner_model_list, data_handler.inner_classic_ANODE_innermodel_datadict_test['loader'], minval, maxval, meanval, stdval, device, verbose=False)
    if extra_signal:
        log_p_data_extrasig = compute_likelihoods(inner_model_list, data_handler.inner_classic_ANODE_innermodel_datadict_extrasig['loader'], minval, maxval, meanval, stdval, device, verbose=False)

    R_ANODE_test = torch.exp(log_p_data_test - log_p_bkg_test).detach().cpu().numpy()
    if extra_signal:
        R_ANODE_extrasig = torch.exp(log_p_data_extrasig - log_p_bkg_extrasig).detach().cpu().numpy()

    preds_matrix = R_ANODE_test.reshape(1,1,-1)
    if extra_signal:
        preds_matrix_extrasig = R_ANODE_extrasig.reshape(1,1,-1)
    else:
        preds_matrix_extrasig = None
    sig_labels = data_handler.inner_classic_ANODE_innermodel_datadict_test['sigorbg'].detach().cpu().numpy()

    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        np.save(os.path.join(savedir, 'preds_matris.npy'), preds_matrix)
        np.save(os.path.join(savedir, 'sig_labels.npy'), sig_labels)
        if extra_signal:
            np.save(os.path.join(savedir, 'preds_matris_extrasig.npy'), preds_matrix_extrasig)

    return preds_matrix, sig_labels, preds_matrix_extrasig
