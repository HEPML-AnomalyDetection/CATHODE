import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from matplotlib import cm
from os.path import join
from data_handler import load_dataset
from density_estimator import DensityEstimator
from evaluation_utils import minimum_validation_loss_models
from scipy.spatial.distance import jensenshannon


def save_to_json(file_name, dictionary):
    with open(file_name, "w") as outfile:
        json.dump(dictionary, outfile)
    print(f"saved as {file_name}")


def chi2(expected_contents, bin_contents, a_b=None, ndof_div=True):
    if a_b is None:
        a = 1 / sum(expected_contents)
        b = 1 / sum(bin_contents)
    else:
        (a, b) = a_b

    if isinstance(a, float):
        a = a*np.ones(len(expected_contents))
    if isinstance(b, float):
        b = b*np.ones(len(expected_contents))

    zero_mask = (expected_contents > 0.)
    if ndof_div:
        ndof = sum(zero_mask) - 1
    else:
        ndof = 1
    numerator = (a[zero_mask]*expected_contents[zero_mask]
                 - b[zero_mask]*bin_contents[zero_mask])**2
    denominator = (a[zero_mask]**2)*expected_contents[zero_mask]
    return sum(numerator / denominator) / ndof


def find_quantile_binning(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))


def run_bkg_sculpting_study(test_data, datadir, clsf_savedir,
                            latent_evaluation, out_file,
                            selection_efficiencies, config_file,
                            model_file_name, num_DE_models, num_clsf_models,
                            n_bins, run_number=0, SR_percentile=True,
                            SR_range=(3.3, 3.7)):

    # prepare classifier data (e.g. mapping to latent space)
    if latent_evaluation:
        outer_traindata = np.load(
            join(datadir, "outerdata_train.npy")
        ).astype("float32")
        outer_traindict = load_dataset(outer_traindata)
        prep_datadict = load_dataset(
            test_data, external_datadict=outer_traindict
        )
        DE_val_losses = np.load(
            join(clsf_savedir, model_file_name+"_val_losses.npy")
        )
        idx = np.argpartition(DE_val_losses, num_DE_models)[:num_DE_models]
        model_path = join(
            clsf_savedir, model_file_name+'_epoch_'+str(idx[0]-1)+'.par'
        )
        model = DensityEstimator(config_file, eval_mode=True,
                                 load_path=model_path).model

        clsf_data = model(
            prep_datadict["tensor2"], prep_datadict["labels"]
        )[0].detach().cpu().numpy()
        SR_mask = np.logical_and(
            (prep_datadict["labels"].detach().cpu().numpy() >= SR_range[0]),
            (prep_datadict["labels"].detach().cpu().numpy() <= SR_range[1]),
        ).flatten()
    else:
        inner_traindata = np.load(
            join(datadir, "innerdata_train.npy")
        ).astype("float32")[:, 1: -1]
        mean = np.mean(inner_traindata, axis=0)
        std = np.std(inner_traindata, axis=0)
        clsf_data = (test_data[:, 1: -1] - mean) / std
        SR_mask = np.logical_and(
            (test_data[:, 0] >= SR_range[0]),
            (test_data[:, 0] <= SR_range[1]),
        ).flatten()

    # get classifier score
    clsf_model_paths = minimum_validation_loss_models(
        clsf_savedir, n_epochs=num_clsf_models
    )[run_number]

    use_cuda = False  # torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    predictions = []
    for model_path in clsf_model_paths:
        clsf_model = torch.load(model_path, map_location=device)
        clsf_model.eval()
        predictions.append(clsf_model.predict(clsf_data).flatten())
    mean_predictions = np.mean(np.stack(predictions, axis=0), axis=0)

    # select X% most anomalous events and plot
    plt.clf()
    if latent_evaluation:
        masked_data = test_data[prep_datadict["mask"].detach().cpu().numpy()]
    else:
        masked_data = test_data

    preds_mask = (~np.isnan(mean_predictions)) & (~np.isinf(mean_predictions))
    masked_data = masked_data[preds_mask]
    mean_predictions = mean_predictions[preds_mask]
    SR_mask = SR_mask[preds_mask]

    results_dict = {}

    _, binning, _ = plt.hist(
        test_data[:, 0], bins=n_bins, color="grey",
        density=True, alpha=0.8, label="all"
    )
    raw_bin_content, _ = np.histogram(test_data[:, 0], bins=binning)
    quantile_binning = find_quantile_binning(test_data[:, 0], n_bins)
    quantile_bin_content, _ = np.histogram(test_data[:, 0], quantile_binning)

    colors = cm.viridis(np.linspace(0., 0.9, len(selection_efficiencies)))
    for i, sel_eff in enumerate(selection_efficiencies):
        if SR_percentile:
            threshold = np.quantile(mean_predictions[SR_mask], 1-sel_eff)
        else:
            threshold = np.quantile(mean_predictions, 1-sel_eff)
        selected_data = masked_data[mean_predictions > threshold]

        sel_bin_content, _ = np.histogram(selected_data[:, 0], bins=binning)
        jsd = jensenshannon(raw_bin_content, sel_bin_content)
        quantile_sel_bin_content, _ = np.histogram(selected_data[:, 0],
                                                   quantile_binning)
        a_b = [
            1 / (sum(sel_eff*quantile_bin_content)
                 * np.diff(quantile_binning)),
            1 / (sum(quantile_sel_bin_content)
                 * np.diff(quantile_binning)),
        ]
        chi2_val = chi2(sel_eff*quantile_bin_content, quantile_sel_bin_content,
                        a_b=a_b, ndof_div=True)
        _ = plt.hist(
            selected_data[:, 0], bins=binning, color=colors[i],
            density=True, histtype="step",
            label=f"{100*sel_eff:.0f}% sel., JSD={jsd:.3f}"
        )
        results_dict[sel_eff] = {
            "JSD": jsd,
            "chi2": chi2_val,
        }
        np.save(join(clsf_savedir, out_file.replace(".pdf",
                                                    f"_sel{sel_eff:.2f}.npy")),
                selected_data[:, 0])

    plt.legend(loc="upper right")
    plt.yscale("log")
    plt.xlabel(r"$m_{jj}$ (TeV)")
    plt.ylabel(r"events (norm.)")
    out_file_path = join(clsf_savedir, out_file)
    plt.savefig(out_file_path)
    print(f"saved as {out_file_path}")

    return results_dict


def statistical_metrics(test_data, n_bins, sel_eff, n_trials):
    raw_bin_content, binning = np.histogram(test_data[:, 0], bins=n_bins)
    jsd_vals = []
    chi2_vals = []

    quantile_binning = find_quantile_binning(test_data[:, 0], n_bins)
    quantile_bin_content, _ = np.histogram(test_data[:, 0], quantile_binning)

    np.random.seed(42)
    for i in range(n_trials):
        selected_mjj = np.random.choice(
            test_data[:, 0], size=int(sel_eff*test_data.shape[0])
        )
        sel_bin_content, _ = np.histogram(selected_mjj, bins=binning)
        jsd_vals.append(jensenshannon(raw_bin_content, sel_bin_content))
        quantile_sel_bin_content, _ = np.histogram(selected_mjj,
                                                   quantile_binning)
        a_b = [
            1 / (sum(sel_eff*quantile_bin_content)
                 * np.diff(quantile_binning)),
            1 / (sum(quantile_sel_bin_content)
                 * np.diff(quantile_binning)),
        ]
        chi2_vals.append(
            chi2(sel_eff*quantile_bin_content, quantile_sel_bin_content,
                 a_b=a_b, ndof_div=True)
        )

    return {"JSD": jsd_vals, "chi2": chi2_vals}


def JSD_summary(datadir, clsf_basedir, json_out_file, pdf_out_file, mode,
                selection_efficiencies, num_clsf_models=10, num_DE_models=1,
                model_file_name="my_ANODE_model",
                config_file="DE_MAF_model.yml", n_bins=100, n_runs=10,
                SR_percentile=True, SR_range=(3.3, 3.7),
                veto_runs=None):

    # load data and preprocess
    test_data = np.vstack((
        np.load(join(datadir, "innerdata_test.npy")),
        np.load(join(datadir, "outerdata_test.npy"))
    )).astype("float32")
    test_data = test_data[test_data[:, -1] == 0]

    if mode == "stats":
        results_dict = {}
        n_trials = 100
        for sel_eff in selection_efficiencies:
            metrics = statistical_metrics(
                test_data, n_bins, sel_eff, n_trials
            )
            results_dict[sel_eff] = {}
            results_dict[sel_eff]["JSD"] = metrics["JSD"]
            results_dict[sel_eff]["chi2"] = metrics["chi2"]

        save_to_json(json_out_file, results_dict)
    else:
        results_dicts = []
        for run in range(1, n_runs+1):
            if veto_runs is not None and run in veto_runs:
                print(f"skipping run {run}...")
                continue
            if mode == "latentCATHODE":
                latent_evaluation = True
                clsf_savedir = clsf_basedir.format(run)
                run_number = 0
            elif mode == "classicCATHODE":
                latent_evaluation = False
                clsf_savedir = clsf_basedir.format(run)
                run_number = 0
            elif mode == "idealizedAD":
                latent_evaluation = False
                clsf_savedir = clsf_basedir
                run_number = run - 1
            else:
                raise NotImplementedError

            results_dicts.append(run_bkg_sculpting_study(
                test_data, datadir, clsf_savedir, latent_evaluation,
                pdf_out_file.format(run), selection_efficiencies, config_file,
                model_file_name, num_DE_models, num_clsf_models, n_bins,
                run_number=run_number, SR_percentile=SR_percentile,
                SR_range=SR_range
            ))

        results_dict = {}
        n_runs_eff = n_runs
        if veto_runs is not None:
            n_runs_eff -= len(veto_runs)
        for sel_eff in selection_efficiencies:
            results_dict[sel_eff] = {}

            jsd_list = [
                results_dicts[i][sel_eff]["JSD"] for i in range(n_runs_eff)
            ]
            results_dict[sel_eff]["JSD"] = jsd_list
            chi2_list = [
                results_dicts[i][sel_eff]["chi2"] for i in range(n_runs_eff)
            ]
            results_dict[sel_eff]["chi2"] = chi2_list

        save_to_json(json_out_file, results_dict)
