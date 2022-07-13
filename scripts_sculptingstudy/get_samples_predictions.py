import numpy as np
import torch
import sys
from sklearn.neighbors import KernelDensity
import argparse
from os.path import join, dirname, abspath
sys.path.insert(0, join(dirname(abspath(__file__)), "../"))
from data_handler import quick_logit, logit_transform_inverse  # noqa: E402
from density_estimator import DensityEstimator  # noqa: E402


def get_traindict(data, mode):

    out_dict = dict()

    # for sideband (i.e. density estimator models) do full preprocessing chain
    # -> normalize -> logit -> standardize
    if mode == "SB":
        out_dict["max"] = np.max(data, axis=0)
        out_dict["min"] = np.min(data, axis=0)
        data = (data - out_dict["min"])/(out_dict["max"]-out_dict["min"])
        mask = np.prod(((data > 0) & (data < 1)), axis=1, dtype="bool")
        data = data[mask]
        data = np.log(data/(1-data))

        for idx in range(data.shape[1]):
            if np.isinf(data[:, idx]).any():
                no_inf_max = np.max(data[~np.isinf(data[:, idx]), idx])
                no_inf_min = np.min(data[~np.isinf(data[:, idx]), idx])
                data[data[:, idx] == np.inf, idx] = no_inf_max
                data[data[:, idx] == -np.inf, idx] = no_inf_min

        out_dict["mean"] = np.mean(data, axis=0)
        out_dict["std"] = np.std(data, axis=0)

        return out_dict

    # for signal region (i.e. classifier models), just run standardization
    elif mode == "SR":
        out_dict["mean"] = np.mean(data, axis=0)
        out_dict["std"] = np.std(data, axis=0)

        return out_dict

    else:
        raise ValueError("mode option must be either SR or SB!")


def produce_samples(data_train_sr, data_train_sb, data_test_sr, DE_model_list,
                    clsf_model_list, n_samples, out_dir):

    raw_train_mjj_vals = data_train_sr[:, 0]

    # fitting and sampling KDE
    mjj_logit = quick_logit(raw_train_mjj_vals)
    KDE_bandwidth = mjj_logit.shape[0]**(-0.4)
    train_mjj_vals = logit_transform_inverse(KernelDensity(
        bandwidth=KDE_bandwidth, kernel='gaussian').fit(
            mjj_logit.reshape(-1, 1)).sample(n_samples),
        max(raw_train_mjj_vals).item(),
        min(raw_train_mjj_vals).item())

    train_mjj_vals = train_mjj_vals.astype(np.float32)
    train_mjj_vals_torch = torch.from_numpy(train_mjj_vals).reshape((-1, 1))
    cond_inputs = train_mjj_vals_torch.reshape((-1, 1))

    # sample auxiliary variables with these conditionals
    outer_traindict = get_traindict(data_train_sb[:, 1:-1], mode="SB")
    inner_traindict = get_traindict(data_train_sr[:, 1:-1], mode="SR")

    sample_list = []
    n_samples_per_model = int(n_samples/len(DE_model_list))
    for i, outer_model in enumerate(DE_model_list):
        print(f"sampling from model {i+1}/{len(DE_model_list)}")
        current_samps_tensor = outer_model.model.sample(
            num_samples=n_samples_per_model,
            cond_inputs=cond_inputs[
                i*n_samples_per_model:(i+1)*n_samples_per_model])

        samples_tmp = current_samps_tensor.detach().cpu().numpy()
        samples_tmp = ((samples_tmp*outer_traindict['std'])
                       + outer_traindict['mean'])
        samples_tmp = 1/(1 + np.exp(-samples_tmp))
        samples_tmp = (
            (samples_tmp*(outer_traindict['max']-outer_traindict['min']))
            + outer_traindict['min'])

        sample_list.append(samples_tmp)

    full_samples = np.concatenate(sample_list)

    # shuffle samples
    shuffled_indices = np.random.permutation(full_samples.shape[0])
    shuffled_samples = full_samples[shuffled_indices]
    cond_inputs = cond_inputs.detach().cpu().numpy()
    shuffled_cond_inputs = cond_inputs[shuffled_indices]
    full_samples = np.hstack((shuffled_cond_inputs, shuffled_samples))

    # store untransformed samples
    np.save(join(out_dir, "samps_full.npy"), full_samples)
    np.save(join(out_dir, "data_full.npy"), data_test_sr)

    # transform samples with inner transformation (just standardize)
    full_samples[:, 1:] = ((full_samples[:, 1:] - inner_traindict["mean"])
                           / inner_traindict["std"])
    data_test_sr[:, 1:-1] = ((data_test_sr[:, 1:-1] - inner_traindict["mean"])
                             / inner_traindict["std"])
    sample_tensor = torch.from_numpy(
        full_samples[:, 1:]).type(torch.FloatTensor)

    full_preds = torch.zeros((data_test_sr.shape[0],))
    full_samps_preds = torch.zeros((sample_tensor.shape[0],))
    with torch.no_grad():
        for inner_model in clsf_model_list:

            tensor_data_test_sr = torch.from_numpy(
                data_test_sr).type(torch.FloatTensor)

            current_preds = inner_model.predict(
                tensor_data_test_sr[:, 1:-1]).flatten()

            current_samps_preds = inner_model.predict(sample_tensor).flatten()
            full_preds = full_preds + current_preds
            full_samps_preds = full_samps_preds + current_samps_preds

        full_preds = full_preds.detach().cpu().numpy()
        full_samps_preds = full_samps_preds.detach().cpu().numpy()
        full_preds = full_preds/len(clsf_model_list)
        full_samps_preds = full_samps_preds/len(clsf_model_list)

    # make cut and store thresholds
    eff_list = [0.2, 0.05, 0.01, 0.001]
    indices = np.argsort(full_preds)
    np.save(join(out_dir, "data_preds_full.npy"), full_preds)
    np.save(join(out_dir, "samps_preds_full.npy"), full_samps_preds)

    for eff in eff_list:
        num_selected = int(np.floor(eff*data_test_sr.shape[0]))
        eff_indices = indices[-num_selected:]
        threshold = full_preds[eff_indices[0]]
        cut_data_mjj = data_test_sr[eff_indices, 0]
        cut_samples_mask = (full_samps_preds > threshold)
        cut_samples_mjj = full_samples[cut_samples_mask, 0]
        np.save(join(out_dir, f"mjj_data_eff{eff}.npy"), cut_data_mjj)
        np.save(join(out_dir, f"mjj_samps_eff{eff}.npy"), cut_samples_mjj)


def find_best_DE_epochs(result_dir, prefix, num_models):
    """ looks through saved val-losses and creates list-of-paths of
    num_models best ones"""
    val_losses = np.load(join(result_dir, prefix + "_val_losses.npy"))
    idx = np.argpartition(val_losses, num_models)[:num_models]
    ret_list = []
    for index in idx:
        ret_list.append(join(result_dir,
                             prefix+'_epoch_'+str(index-1)+'.par'))
    return ret_list


def find_best_clsf_epochs(result_dir, prefix, num_models):
    """ looks through saved val-losses and creates list-of-paths of
    num_models best ones"""

    # force run 0
    val_losses = np.load(join(result_dir, "val_loss_matris.npy"))[0]
    idx = np.argpartition(val_losses, num_models)[:num_models]
    ret_list = []
    for index in idx:
        if index == 0:
            my_index = 1
        else:
            my_index = index-1

        ret_list.append(join(result_dir,
                             prefix+'_run0_ep' + str(my_index)))
    return ret_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            """
            Select m_jj values based on selection efficiency of cut on CATHODE
            tagger
            """),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=str, required=True,
                        help="""Directory where data is stored""")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="""Directory where selected events should be
                        stored""")
    parser.add_argument("--result-clsf-dir", type=str, required=True,
                        help="""Directory containing training results""")
    parser.add_argument("--DE-config", type=str, required=True,
                        help="""Path to density estimator config YAML file.""")
    parser.add_argument("--result-DE-dir", type=str, required=True,
                        help="""Directory containing training results""")

    args = parser.parse_args()

    # selecting appropriate device
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda:0" if CUDA else "cpu")

    data_train_sr = np.load(join(args.data_dir, "innerdata_train.npy"))
    data_train_sb = np.load(join(args.data_dir, "outerdata_train.npy"))
    data_test_sr = np.load(join(args.data_dir, "innerdata_test.npy"))
    DE_model_files = find_best_DE_epochs(args.result_DE_dir, "my_ANODE_model",
                                         10)

    clsf_model_files = find_best_clsf_epochs(args.result_clsf_dir, "model", 1)
    DE_model_list = []

    for DE_file in DE_model_files:
        tmp_model = DensityEstimator(args.DE_config,
                                     eval_mode=True,
                                     load_path=DE_file,
                                     device=device, verbose=False,
                                     bound=False)
        DE_model_list.append(tmp_model)

    clsf_model_list = [
        torch.load(model_path,
                   map_location=device) for model_path in clsf_model_files]

    produce_samples(data_train_sr, data_train_sb, data_test_sr, DE_model_list,
                    clsf_model_list, data_test_sr.shape[0], args.out_dir)

    print("Done!")
