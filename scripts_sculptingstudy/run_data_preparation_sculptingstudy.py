import pandas as pd
import numpy as np
import os
import argparse
import vector
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
    description=("Prepare LHCO dataset for sculpting study."),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--outdir", type=str, default="preprocessed_data/",
                    help="output directory")
parser.add_argument("--extra_bkg", action="store_true", default=False,
                    help="run on extra background sample instead")
parser.add_argument("--S_over_B", type=float, default=-1,
                    help="Signal over background ratio in the signal region.")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed for the mixing")
parser.add_argument("--sr_limits", nargs=2, type=float, required=True,
                    help="Upper and lower mjj limits for the signal region "
                    "in TeV")
parser.add_argument("--split_data_more", action="store_true", default=False,
                    help=("Split the test data already into validation "
                          "and test data"))
parser.add_argument("--add_deltar", action="store_true", default=False,
                    help=("Include delta_r between the two jets "
                          "as input feature"))
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

if args.extra_bkg:
    features = pd.read_hdf(("events_anomalydetection_DelphesPythia8_v2_qcd"
                            "_extra_inneronly_batch2_features.h5"))
else:
    # This is from the R&D Zenodo https://doi.org/10.5281/zenodo.2629072
    features = pd.read_hdf(
        ("/beegfs/desy/user/loeschet/american_cathode/"
         "CATHODE/events_anomalydetection_v2.features.h5"))

if args.extra_bkg:
    features_bg = features
else:
    features_sig = features[features['label'] == 1]
    features_bg = features[features['label'] == 0]

# Read from data
mj1mj2_bg = np.array(features_bg[['mj1', 'mj2']])
tau21_bg = (np.array(features_bg[['tau2j1', 'tau2j2']])
            / (1e-5+np.array(features_bg[['tau1j1', 'tau1j2']])))

if not args.extra_bkg:
    mj1mj2_sig = np.array(features_sig[['mj1', 'mj2']])
    tau21_sig = (np.array(features_sig[['tau2j1', 'tau2j2']])
                 / (1e-5+np.array(features_sig[['tau1j1', 'tau1j2']])))

# Sorting of mj1 and mj2:
# Identifies which column has the minimum of mj1 and mj2, and sorts it so
# the new array mjmin contains the mj with the smallest energy, and mjmax is
# the one with the biggest.
mjmin_bg = mj1mj2_bg[range(len(mj1mj2_bg)), np.argmin(mj1mj2_bg, axis=1)]
mjmax_bg = mj1mj2_bg[range(len(mj1mj2_bg)), np.argmax(mj1mj2_bg, axis=1)]

if not args.extra_bkg:
    mjmin_sig = mj1mj2_sig[range(len(mj1mj2_sig)),
                           np.argmin(mj1mj2_sig, axis=1)]
    mjmax_sig = mj1mj2_sig[range(len(mj1mj2_sig)),
                           np.argmax(mj1mj2_sig, axis=1)]

# Then we do the same sorting for the taus
tau21min_bg = tau21_bg[range(len(mj1mj2_bg)), np.argmin(mj1mj2_bg, axis=1)]
tau21max_bg = tau21_bg[range(len(mj1mj2_bg)), np.argmax(mj1mj2_bg, axis=1)]

if not args.extra_bkg:
    tau21min_sig = tau21_sig[range(len(mj1mj2_sig)),
                             np.argmin(mj1mj2_sig, axis=1)]
    tau21max_sig = tau21_sig[range(len(mj1mj2_sig)),
                             np.argmax(mj1mj2_sig, axis=1)]

# Calculate mjj and collect the features into a dataset,
# plus mark signal/bg with 1/0
if not args.extra_bkg:
    pjj_sig = (np.array(features_sig[['pxj1', 'pyj1', 'pzj1']])
               + np.array(features_sig[['pxj2', 'pyj2', 'pzj2']]))

    Ejj_sig = (
        np.sqrt(np.sum(
            np.array(features_sig[['pxj1', 'pyj1', 'pzj1', 'mj1']])**2, axis=1)
                )
        + np.sqrt(np.sum(
            np.array(features_sig[['pxj2', 'pyj2', 'pzj2', 'mj2']])**2, axis=1)
                  ))

    mjj_sig = np.sqrt(Ejj_sig**2-np.sum(pjj_sig**2, axis=1))

    # compute DeltaR JJ
    j1_vec_sig = vector.array({
        "px": np.array(features_sig[["pxj1"]]),
        "py": np.array(features_sig[["pyj1"]]),
        "pz": np.array(features_sig[["pzj1"]]),
    })

    j2_vec_sig = vector.array({
        "px": np.array(features_sig[["pxj2"]]),
        "py": np.array(features_sig[["pyj2"]]),
        "pz": np.array(features_sig[["pzj2"]]),
    })

    deltaR_jj_sig = j1_vec_sig.deltaR(j2_vec_sig)
    deltaeta_jj_sig = j1_vec_sig.deltaeta(j2_vec_sig)

# compute DeltaR JJ
j1_vec_bg = vector.array({
    "px": np.array(features_bg[["pxj1"]]),
    "py": np.array(features_bg[["pyj1"]]),
    "pz": np.array(features_bg[["pzj1"]]),
})

j2_vec_bg = vector.array({
    "px": np.array(features_bg[["pxj2"]]),
    "py": np.array(features_bg[["pyj2"]]),
    "pz": np.array(features_bg[["pzj2"]]),
})

deltaR_jj_bg = j1_vec_bg.deltaR(j2_vec_bg)
deltaeta_jj_bg = j1_vec_bg.deltaeta(j2_vec_bg)

pjj_bg = (
    (np.array(features_bg[['pxj1', 'pyj1', 'pzj1']])
     + np.array(features_bg[['pxj2', 'pyj2', 'pzj2']]))
    )

Ejj_bg = (
    np.sqrt(
        np.sum(
            np.array(features_bg[['pxj1', 'pyj1', 'pzj1', 'mj1']])**2, axis=1))
    + np.sqrt(
        np.sum(
            np.array(features_bg[['pxj2', 'pyj2', 'pzj2', 'mj2']])**2, axis=1))
    )

mjj_bg = np.sqrt(Ejj_bg**2-np.sum(pjj_bg**2, axis=1))

if args.add_deltar:
    input_list_bg = [mjj_bg/1000, mjmin_bg/1000, (mjmax_bg-mjmin_bg)/1000,
                     tau21min_bg, tau21max_bg, deltaR_jj_bg.flatten(),
                     np.zeros(len(mjj_bg))]
else:
    input_list_bg = [mjj_bg/1000, mjmin_bg/1000, (mjmax_bg-mjmin_bg)/1000,
                     tau21min_bg, tau21max_bg, np.zeros(len(mjj_bg))]

dataset_bg = np.dstack(input_list_bg)[0]

if not args.extra_bkg:
    if args.add_deltar:
        input_list_sig = [mjj_sig/1000, mjmin_sig/1000,
                          (mjmax_sig-mjmin_sig)/1000, tau21min_sig,
                          tau21max_sig, deltaR_jj_sig.flatten(),
                          np.zeros(len(mjj_sig))]
    else:
        input_list_sig = [mjj_sig/1000, mjmin_sig/1000,
                          (mjmax_sig-mjmin_sig)/1000,
                          tau21min_sig, tau21max_sig,
                          np.zeros(len(mjj_sig))]

        dataset_sig = np.dstack(input_list_sig)[0]

np.random.seed(args.seed)  # Set random seed so we get a deterministic result

if args.seed != 1 and not args.extra_bkg:
    np.random.shuffle(dataset_sig)

if args.S_over_B == -1:
    n_sig = 1000
else:
    n_sig = int(args.S_over_B*1000/0.006361658645922605)

if args.extra_bkg or (args.S_over_B == 0):
    data_all = dataset_bg
else:
    data_all = np.concatenate((dataset_bg, dataset_sig[:n_sig]))
indices = np.array(range(len(data_all))).astype('int')
np.random.shuffle(indices)

data_all = data_all[indices]
# format of data_all: mjj (TeV), mjmin (TeV), mjmax-mjmin (TeV), tau21(mjmin),
# tau21 (mjmax), sigorbg label

sr_limits = np.sort(args.sr_limits)
minmass = sr_limits[0]
maxmass = sr_limits[1]

innermask = (data_all[:, 0] > minmass) & (data_all[:, 0] < maxmass)
outermask = ~innermask

innerdata = data_all[innermask]
outerdata = data_all[outermask]

if args.extra_bkg:
    if args.split_data_more:
        # split extra background into train and test data
        n_train = int(60000./122124*innerdata.shape[0])
        innerdata_train = innerdata[:n_train]
        innerdata_test = innerdata[n_train:]

        # split extra background test data into test and validation
        n_val = int(2/5*innerdata_test.shape[0])
        innerdata_val = innerdata_test[:n_val]
        innerdata_test = innerdata_test[n_val:]
        np.save(os.path.join(args.outdir, 'innerdata_extrabkg_train.npy'),
                innerdata_train)
        np.save(os.path.join(args.outdir, 'innerdata_extrabkg_val.npy'),
                innerdata_val)
        np.save(os.path.join(args.outdir, 'innerdata_extrabkg_test.npy'),
                innerdata_test)
    else:
        np.save(os.path.join(args.outdir, 'innerdata_extrabkg_test.npy'),
                innerdata)
else:
    # train_size parameters have been hard coded to reflect the
    # train/val/test splittings used in the original CATHODE paper
    outerdata_train, outerdata_test = train_test_split(outerdata,
                                                       train_size=0.57)

    innerdata_train, innerdata_valtest = train_test_split(innerdata,
                                                          train_size=0.5)

    if args.S_over_B != 0:
        innerdata_extrasig = dataset_sig[n_sig:]
        innerdata_extrasig = innerdata_extrasig[
            (innerdata_extrasig[:, 0] > minmass)
            & (innerdata_extrasig[:, 0] < maxmass)]

    if args.split_data_more:
        innerdata_val, innerdata_test = train_test_split(innerdata_valtest,
                                                         train_size=0.4)

        np.save(os.path.join(args.outdir, 'innerdata_val.npy'), innerdata_val)

        if args.S_over_B != 0:

            # splitting extra signal into train and test set
            n_train = int(60000./122124*innerdata_extrasig.shape[0])
            innerdata_extrasig_train = innerdata_extrasig[:n_train]
            innerdata_extrasig_test = innerdata_extrasig[n_train:]
            n_val = int(2/5*innerdata_extrasig_test.shape[0])
            innerdata_extrasig_val = innerdata_extrasig_test[:n_val]
            innerdata_extrasig_test = innerdata_extrasig_test[n_val:]
            np.save(os.path.join(args.outdir, 'innerdata_extrasig_train.npy'),
                    innerdata_extrasig_train)
            np.save(os.path.join(args.outdir, 'innerdata_extrasig_val.npy'),
                    innerdata_extrasig_val)
            np.save(os.path.join(args.outdir, 'innerdata_extrasig_test.npy'),
                    innerdata_extrasig_test)

    else:
        innerdata_test = innerdata_valtest
        if args.S_over_B != 0:
            np.save(os.path.join(args.outdir, 'innerdata_extrasig.npy'),
                    innerdata_extrasig)

    np.save(os.path.join(args.outdir, 'innerdata_test.npy'), innerdata_test)
    np.save(os.path.join(args.outdir, 'outerdata_train.npy'), outerdata_train)
    np.save(os.path.join(args.outdir, 'innerdata_train.npy'), innerdata_train)
    np.save(os.path.join(args.outdir, 'outerdata_test.npy'), outerdata_test)
print("saved in "+args.outdir)
