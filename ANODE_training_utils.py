import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from tqdm import tqdm
import flows as fnn

def train_ANODE(model, optimizer, dataloader_train, dataloader_test, model_file_name, epochs,
                savedir="ANODE_models/", device=torch.device('cpu'), verbose=True,
                no_logit=False, data_std=None):
    # ANODE model training function. Records training and valdiation losses and saves the
    #   parameters to file. Works for either inner or outer model.
    #   no_logit corrects printed and saved loss value for not using logit transformation

    # Create model directory if necessary
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Record also untrained losses

    if no_logit:
        assert (data_std is not None), (
            "Need data_std to correct losses when trained without logit")

    train_loss_return = compute_loss_over_batches(model, dataloader_train, device,
                                                  correct_logit=data_std if no_logit else None)
    val_loss_return = compute_loss_over_batches(model, dataloader_test, device,
                                                correct_logit=data_std if no_logit else None)
    train_loss = train_loss_return[0]
    val_loss = val_loss_return[0]
    if no_logit:
        print("uncorrected train_loss = ", train_loss_return[1])
        print("uncorrected val_loss = ", val_loss_return[1])
    print("train_loss = ", train_loss)
    print("val_loss = ", val_loss)
    train_losses = 1e20*np.ones(epochs+1, dtype=np.float32)
    val_losses = 1e20*np.ones(epochs+1, dtype=np.float32)
    train_losses[0] = train_loss
    val_losses[0] = val_loss
    np.save(os.path.join(savedir, model_file_name+"_train_losses.npy"), train_losses)
    np.save(os.path.join(savedir, model_file_name+"_val_losses.npy"), val_losses)

    # Actually train model
    for epoch in range(epochs):
        print('\nEpoch: {}'.format(epoch))
        train_loss_return = train_epoch(model, optimizer, dataloader_train, device, verbose=verbose,
                                        data_std=data_std if no_logit else None)
        #train_loss_return = compute_loss_over_batches(model, dataloader_train, device,
        #                                              correct_logit=data_std if no_logit else None)
        val_loss_return = compute_loss_over_batches(model, dataloader_test, device,
                                                    correct_logit=data_std if no_logit else None)
        train_loss = train_loss_return[0]
        val_loss = val_loss_return[0]
        if no_logit:
            print("uncorrected train_loss = ", train_loss_return[1])
            print("uncorrected val_loss = ", val_loss_return[1])
        print("train_loss = ", train_loss)
        print("val_loss = ", val_loss)
        train_losses[epoch+1] = train_loss
        val_losses[epoch+1] = val_loss
        np.save(os.path.join(savedir, model_file_name+"_train_losses.npy"), train_losses)
        np.save(os.path.join(savedir, model_file_name+"_val_losses.npy"), val_losses)
        torch.save(model.state_dict(), os.path.join(savedir,
                                                    model_file_name+"_epoch_"+str(epoch)+".par"))


def train_epoch(model, optimizer, data_loader, device, verbose=True, data_std=None):
    # Does one epoch of ANODE model training.

    model.train()
    train_loss = 0
    train_loss_avg = []
    if data_std is not None:
        corrected_train_loss_avg = []
    if verbose:
        pbar = tqdm(total=len(data_loader.dataset))
    for batch_idx, data in enumerate(data_loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)

        optimizer.zero_grad()
        loss = -model.log_probs(data, cond_data)
        train_loss += loss.mean().item()
        train_loss_avg.extend(loss.tolist())
        if data_std is not None:
            mask = (data > 0) & (data < 1)
            data = data[mask.all(dim=1)]
            corrected_loss = (loss[mask.all(dim=1)].flatten() -\
                torch.log(data_std*data*(1.-data)).sum(dim=1)).flatten()
            corrected_train_loss_avg.extend(corrected_loss.tolist())
        loss.mean().backward()
        optimizer.step()

        if verbose:
            pbar.update(data.size(0))
            pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
                -train_loss / (batch_idx + 1)))

    if verbose:
        pbar.close()

    has_batch_norm = False
    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            has_batch_norm = True
            module.momentum = 0

    if has_batch_norm:
        with torch.no_grad():
            ## NOTE this is not yet fully understood but it crucial to work with BN
            model(data_loader.dataset.tensors[0].to(data.device),
                  data_loader.dataset.tensors[1].to(data.device).float())

        for module in model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 1
    if data_std is not None:
        return (np.array(corrected_train_loss_avg).flatten().mean(),
                np.array(train_loss_avg).flatten().mean())
    else:
        return (np.array(train_loss_avg).flatten().mean(), )


def compute_loss_over_batches(model, data_loader, device, correct_logit=None):
    # for computing the averaged loss over the entire dataset.
    # Mainly useful for tracking losses during training
    model.eval()
    with torch.no_grad():
        if correct_logit is not None:
            corrected_now_loss = 0.
        now_loss = 0
        n_nans = 0
        n_highs = 0
        for batch_idx, batch_data in enumerate(data_loader):

            data = batch_data[0]
            data = data.to(device)
            cond_data = batch_data[1].float()
            cond_data = cond_data.to(device)

            loss_vals_raw = model.log_probs(data, cond_data)
            loss_vals = loss_vals_raw.flatten()

            if correct_logit is not None:
                mask = (data > 0) & (data < 1)
                data = data[mask.all(dim=1)]
                loss_vals_raw = loss_vals_raw[mask.all(dim=1)]
                corrected_loss_vals_raw = loss_vals_raw.flatten() +\
                    torch.log(correct_logit*data*(1.-data)).sum(dim=1)
                corrected_loss_vals = corrected_loss_vals_raw.flatten()

            n_nans += sum(torch.isnan(loss_vals)).item()
            n_highs += sum(torch.abs(loss_vals) >= 1000).item()
            loss_vals = loss_vals[~torch.isnan(loss_vals)]
            loss_vals = loss_vals[torch.abs(loss_vals) < 1000]
            loss = -loss_vals.mean()
            loss = loss.item()

            if correct_logit is not None:
                corrected_now_loss -= corrected_loss_vals.mean().item()
                corrected_end_loss = corrected_now_loss / (batch_idx + 1)

            now_loss += loss
            end_loss = now_loss / (batch_idx + 1)
        print("n_nans =", n_nans)
        print("n_highs =", n_highs)
        if correct_logit is not None:
            return (corrected_end_loss, end_loss)
        else:
            return (end_loss, )

def plot_ANODE_losses(train_losses, val_losses, yrange=None, savefig=None, suppress_show=False):
    # plots the ANODE training losses from loss array. The image is saved if a filename is
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
    plt.ylabel("(Mean) Negative Log Likelihood Loss")
    plt.legend(loc="upper right", frameon=False)
    if savefig is not None:
        plt.savefig(savefig+".pdf", bbox_inches="tight")
    if not suppress_show:
        plt.show()
    plt.close()
