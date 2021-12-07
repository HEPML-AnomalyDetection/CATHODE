import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(
    description=("Prepare LHCO dataset."),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--outdir", type=str, default="preprocessed_data/",
                    help="output directory")
parser.add_argument("--S_over_B", type=float, default=-1,
                    help="Signal over background ratio in the signal region.")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed for the mixing")
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

# the "data" containing too much signal
features=pd.read_hdf("events_anomalydetection_v2.features.h5")

# additionally produced bkg
features_extrabkg = pd.read_hdf("events_anomalydetection_qcd_extra_inneronly_features.h5")

## to be split among the different sets 
features_extrabkg1 = features_extrabkg[:312858]

## to be used to enhance the evalaution
features_extrabkg2 = features_extrabkg[312858:]

features_sig=features[features['label']==1]
features_bg=features[features['label']==0]

# Read from data
mj1mj2_bg = np.array(features_bg[['mj1','mj2']])
tau21_bg = np.array(features_bg[['tau2j1','tau2j2']])/(1e-5+np.array(features_bg[['tau1j1','tau1j2']]))
mj1mj2_sig = np.array(features_sig[['mj1','mj2']])
tau21_sig = np.array(features_sig[['tau2j1','tau2j2']])/(1e-5+np.array(features_sig[['tau1j1','tau1j2']]))
mj1mj2_extrabkg1 = np.array(features_extrabkg1[['mj1','mj2']])
tau21_extrabkg1 = np.array(features_extrabkg1[['tau2j1','tau2j2']])/(1e-5+np.array(features_extrabkg1[['tau1j1','tau1j2']]))
mj1mj2_extrabkg2 = np.array(features_extrabkg2[['mj1','mj2']])
tau21_extrabkg2 = np.array(features_extrabkg2[['tau2j1','tau2j2']])/(1e-5+np.array(features_extrabkg2[['tau1j1','tau1j2']]))


# Sorting of mj1 and mj2:
# Identifies which column has the minimum of mj1 and mj2, and sorts it so the new array mjmin contains the 
# mj with the smallest energy, and mjmax is the one with the biggest.
mjmin_bg = mj1mj2_bg[range(len(mj1mj2_bg)), np.argmin(mj1mj2_bg, axis=1)] 
mjmax_bg = mj1mj2_bg[range(len(mj1mj2_bg)), np.argmax(mj1mj2_bg, axis=1)]
mjmin_sig = mj1mj2_sig[range(len(mj1mj2_sig)), np.argmin(mj1mj2_sig, axis=1)]
mjmax_sig = mj1mj2_sig[range(len(mj1mj2_sig)), np.argmax(mj1mj2_sig, axis=1)]
mjmin_extrabkg1 = mj1mj2_extrabkg1[range(len(mj1mj2_extrabkg1)), np.argmin(mj1mj2_extrabkg1, axis=1)] 
mjmax_extrabkg1 = mj1mj2_extrabkg1[range(len(mj1mj2_extrabkg1)), np.argmax(mj1mj2_extrabkg1, axis=1)]
mjmin_extrabkg2 = mj1mj2_extrabkg2[range(len(mj1mj2_extrabkg2)), np.argmin(mj1mj2_extrabkg2, axis=1)] 
mjmax_extrabkg2 = mj1mj2_extrabkg2[range(len(mj1mj2_extrabkg2)), np.argmax(mj1mj2_extrabkg2, axis=1)]

# Then we do the same sorting for the taus
tau21min_bg=tau21_bg[range(len(mj1mj2_bg)), np.argmin(mj1mj2_bg, axis=1)]
tau21max_bg=tau21_bg[range(len(mj1mj2_bg)), np.argmax(mj1mj2_bg, axis=1)]
tau21min_sig=tau21_sig[range(len(mj1mj2_sig)), np.argmin(mj1mj2_sig, axis=1)]
tau21max_sig=tau21_sig[range(len(mj1mj2_sig)), np.argmax(mj1mj2_sig, axis=1)]
tau21min_extrabkg1 = tau21_extrabkg1[range(len(mj1mj2_extrabkg1)), np.argmin(mj1mj2_extrabkg1, axis=1)]
tau21max_extrabkg1 = tau21_extrabkg1[range(len(mj1mj2_extrabkg1)), np.argmax(mj1mj2_extrabkg1, axis=1)]
tau21min_extrabkg2 = tau21_extrabkg2[range(len(mj1mj2_extrabkg2)), np.argmin(mj1mj2_extrabkg2, axis=1)]
tau21max_extrabkg2 = tau21_extrabkg2[range(len(mj1mj2_extrabkg2)), np.argmax(mj1mj2_extrabkg2, axis=1)]


# Calculate mjj and collect the features into a dataset, plus mark signal/bg with 1/0
pjj_sig = (np.array(features_sig[['pxj1','pyj1','pzj1']])+np.array(features_sig[['pxj2','pyj2','pzj2']]))
Ejj_sig = np.sqrt(np.sum(np.array(features_sig[['pxj1','pyj1','pzj1','mj1']])**2, axis=1))\
    +np.sqrt(np.sum(np.array(features_sig[['pxj2','pyj2','pzj2','mj2']])**2, axis=1))
mjj_sig = np.sqrt(Ejj_sig**2-np.sum(pjj_sig**2, axis=1))

pjj_bg = (np.array(features_bg[['pxj1','pyj1','pzj1']])+np.array(features_bg[['pxj2','pyj2','pzj2']]))
Ejj_bg = np.sqrt(np.sum(np.array(features_bg[['pxj1','pyj1','pzj1','mj1']])**2, axis=1))\
    +np.sqrt(np.sum(np.array(features_bg[['pxj2','pyj2','pzj2','mj2']])**2, axis=1))
mjj_bg = np.sqrt(Ejj_bg**2-np.sum(pjj_bg**2, axis=1))

pjj_extrabkg1 = (np.array(features_extrabkg1[['pxj1','pyj1','pzj1']])+np.array(features_extrabkg1[['pxj2','pyj2','pzj2']]))
Ejj_extrabkg1 = np.sqrt(np.sum(np.array(features_extrabkg1[['pxj1','pyj1','pzj1','mj1']])**2, axis=1))\
    +np.sqrt(np.sum(np.array(features_extrabkg1[['pxj2','pyj2','pzj2','mj2']])**2, axis=1))
mjj_extrabkg1 = np.sqrt(Ejj_extrabkg1**2-np.sum(pjj_extrabkg1**2, axis=1))
pjj_extrabkg2 = (np.array(features_extrabkg2[['pxj1','pyj1','pzj1']])+np.array(features_extrabkg2[['pxj2','pyj2','pzj2']]))
Ejj_extrabkg2 = np.sqrt(np.sum(np.array(features_extrabkg2[['pxj1','pyj1','pzj1','mj1']])**2, axis=1))\
    +np.sqrt(np.sum(np.array(features_extrabkg2[['pxj2','pyj2','pzj2','mj2']])**2, axis=1))
mjj_extrabkg2 = np.sqrt(Ejj_extrabkg2**2-np.sum(pjj_extrabkg2**2, axis=1))


dataset_bg = np.dstack((mjj_bg/1000, mjmin_bg/1000, (mjmax_bg-mjmin_bg)/1000, tau21min_bg, tau21max_bg, np.zeros(len(mjj_bg))))[0]
dataset_sig = np.dstack((mjj_sig/1000, mjmin_sig/1000, (mjmax_sig-mjmin_sig)/1000, tau21min_sig, tau21max_sig, np.ones(len(mjj_sig))))[0]
dataset_extrabkg1 = np.dstack((mjj_extrabkg1/1000, mjmin_extrabkg1/1000, (mjmax_extrabkg1-mjmin_extrabkg1)/1000, tau21min_extrabkg1, tau21max_extrabkg1, np.zeros(len(mjj_extrabkg1))))[0]
dataset_extrabkg2 = np.dstack((mjj_extrabkg2/1000, mjmin_extrabkg2/1000, (mjmax_extrabkg2-mjmin_extrabkg2)/1000, tau21min_extrabkg2, tau21max_extrabkg2, np.zeros(len(mjj_extrabkg2))))[0]


np.random.seed(args.seed) # Set the random seed so we get a deterministic result

if args.seed!=1:
    np.random.shuffle(dataset_sig)

if args.S_over_B==-1:
    n_sig = 1000
else:
    n_sig = int(args.S_over_B*1000/0.006361658645922605)

data_all = np.concatenate((dataset_bg, dataset_sig[:n_sig]))
indices = np.array(range(len(data_all))).astype('int')
np.random.shuffle(indices)
data_all = data_all[indices]
indices_extrabkg1 = np.array(range(len(dataset_extrabkg1))).astype('int')
np.random.shuffle(indices_extrabkg1)
dataset_extrabkg1 = dataset_extrabkg1[indices_extrabkg1]
indices_extrabkg2 = np.array(range(len(dataset_extrabkg2))).astype('int')
np.random.shuffle(indices_extrabkg2)
dataset_extrabkg2 = dataset_extrabkg2[indices_extrabkg2]

# format of data_all: mjj (TeV), mjmin (TeV), mjmax-mjmin (TeV), tau21(mjmin), tau21 (mjmax), sigorbg label

minmass=3.3
maxmass=3.7

innermask = (data_all[:,0]>minmass) & (data_all[:,0]<maxmass)
outermask = ~innermask
innerdata = data_all[innermask]
outerdata = data_all[outermask]

innermask_extrabkg1 = (dataset_extrabkg1[:,0]>minmass) & (dataset_extrabkg1[:,0]<maxmass)
innerdata_extrabkg1 = dataset_extrabkg1[innermask_extrabkg1]
innermask_extrabkg2 = (dataset_extrabkg2[:,0]>minmass) & (dataset_extrabkg2[:,0]<maxmass)
innerdata_extrabkg2 = dataset_extrabkg2[innermask_extrabkg2]


outerdata_train = outerdata[:500000]
outerdata_val = outerdata[500000:]

innerdata_train = innerdata[:60000]
innerdata_val = innerdata[60000:120000]

innerdata_extrasig = dataset_sig[n_sig:]
innerdata_extrasig = innerdata_extrasig[(innerdata_extrasig[:,0]>minmass) & (innerdata_extrasig[:,0]<maxmass)]

## splitting extra signal into train, val and test set
n_sig_test = 20000
n_extrasig_train =  (innerdata_extrasig.shape[0]-n_sig_test)//2
innerdata_extrasig_test = innerdata_extrasig[:n_sig_test]
innerdata_extrasig_train = innerdata_extrasig[n_sig_test:n_sig_test+n_extrasig_train]
innerdata_extrasig_val = innerdata_extrasig[n_sig_test+n_extrasig_train:]

## splitting extra bkg (1) into train, val and test set
n_bkg_test = 40000
n_extrabkg_train =  (innerdata_extrabkg1.shape[0]-n_bkg_test)//2
innerdata_extrabkg1_test = innerdata_extrabkg1[:n_bkg_test]
innerdata_extrabkg1_train = innerdata_extrabkg1[n_bkg_test:n_bkg_test+n_extrabkg_train]
innerdata_extrabkg1_val = innerdata_extrabkg1[n_bkg_test+n_extrabkg_train:]

## putting together artificial test set
innerdata_test = np.vstack((innerdata_extrabkg1_test, innerdata_extrasig_test))

np.save(os.path.join(args.outdir, 'outerdata_train.npy'), outerdata_train)
np.save(os.path.join(args.outdir, 'outerdata_test.npy'), outerdata_val)
np.save(os.path.join(args.outdir, 'innerdata_train.npy'), innerdata_train)
np.save(os.path.join(args.outdir, 'innerdata_val.npy'), innerdata_val)   
np.save(os.path.join(args.outdir, 'innerdata_test.npy'), innerdata_test)      
np.save(os.path.join(args.outdir, 'innerdata_extrasig_train.npy'), innerdata_extrasig_train)
np.save(os.path.join(args.outdir, 'innerdata_extrasig_val.npy'), innerdata_extrasig_val)
np.save(os.path.join(args.outdir, 'innerdata_extrabkg_train.npy'), innerdata_extrabkg1_train)
np.save(os.path.join(args.outdir, 'innerdata_extrabkg_val.npy'), innerdata_extrabkg1_val)
np.save(os.path.join(args.outdir, 'innerdata_extrabkg_test.npy'), innerdata_extrabkg2)

print("saved in "+args.outdir)
