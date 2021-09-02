import argparse
import os
import torch
import numpy as np
from data_handler import LHCORD_data_handler, sample_handler, mix_data_samples, plot_data_sample_comparison
from density_estimator import DensityEstimator

parser = argparse.ArgumentParser(
    description='Preprocess and mix data for classifier training.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--savedir', type=str, default="classifier_data/",
                    help='Path to directory where processed classifier data will be stored.')
parser.add_argument('--datashift', type=float, default=0.,
                    help='Shift on jet mass variables to be applied on data.')
parser.add_argument('--data_dir', type=str, default="./",
                    help='Path to the input data.')
parser.add_argument('--random_shift', action="store_true", default=False,
                    help='Shift is not correlated to the actual mjj but randomized.')
parser.add_argument('--fiducial_cut', action="store_true", default=False,
                    help='Whether to apply an (ANODE paper) fiducial cut on the data.')
parser.add_argument('--ANODE_models', type=str, nargs="+", default="",
                    help='Space-separated list of ANODE model paths to be used for (ensemble) '
                    'sampling.')
parser.add_argument('--config_file', type=str, default="ANODE_model.yml",
                    help='ANODE model config file (.yml).')
parser.add_argument('--n_samples', type=int, default=130000,
                    help='Number of samples to be generated. Currently the samples will be '
                    'cut down to match data proportion.')
parser.add_argument('--realistic_conditional', action="store_true", default=False,
                    help='Sample the conditional from a KDE fit rather than a uniform distribution.')
parser.add_argument('--KDE_bandwidth', type=float, default=0.01,
                    help='Bandwith of the KDE fit (used when realistic_conditional is selected)')
parser.add_argument('--oversampling', action="store_true", default=False,
                    help='Add the full number of samples to the training set rather than mixing '
                    'it in equal parts with data.')
parser.add_argument('--no_extra_signal', action="store_true", default=False,
                    help='Suppress the processing of the extra signal sample.')
parser.add_argument('--extra_bkg', action="store_true", default=False,
                    help='Make use of extra background (for supervised and idealized AD).')
parser.add_argument('--CWoLa', action="store_true", default=False,
                    help='Use sideband data instead of samples.')
parser.add_argument('--supervised', action="store_true", default=False,
                    help=('Apply splitting into train, validation and test set for fair '
                          'supervised comparison.'))
parser.add_argument('--idealized_AD', action="store_true", default=False,
                    help=('Prepares data for an idealized anomaly detector, letting a classifier'
                          ' distinguish SR data from SR bkg-only data'))
parser.add_argument('--no_logit', action="store_true", default=False,
                    help='Turns off logit tranform.')
parser.add_argument('--no_logit_trained', action="store_true", default=False,
                    help='Use if the ANODE model was trained without logit transformation.')
parser.add_argument('--external_samples', type=str, nargs="+", default="",
                    help='Space-separated list of pre-sampled npy files of physical variables if '
                    'the sampling has been done externally. The format is '
                    '(mjj, mj1, dmj, tau21_1, tau21_2)')
parser.add_argument('--SR_min', type=float, default=3.3,
                    help='Lower boundary of signal region.')
parser.add_argument('--SR_max', type=float, default=3.7,
                    help='Upper boundary of signal region.')
parser.add_argument('--separate_val_set', action="store_true", default=False,
                    help='Define a separate validation set to pick the classifier epochs.')
parser.add_argument('--verbose', action="store_true", default=False,
                    help='Enables more printout.')


@torch.no_grad()
def main(args):

    assert not ((not (args.supervised or args.idealized_AD or args.CWoLa) and\
                 args.external_samples == "") and args.ANODE_models == ""), (
                     "ANODE models need to be given unless CWoLa, supervised, idealized_AD or"
                     " external sampling is used.")

    # selecting appropriate device
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda:0" if CUDA else "cpu")

    # checking for data separation
    data_files = os.listdir(args.data_dir)
    if "innerdata_val.npy" in data_files:
        finer_data_split = True
    else:
        finer_data_split = False

    if finer_data_split:
        innerdata_train_path = [os.path.join(args.data_dir, 'innerdata_train.npy')]
        innerdata_val_path = [os.path.join(args.data_dir, 'innerdata_val.npy')]
        innerdata_test_path = [os.path.join(args.data_dir, 'innerdata_test.npy')]
        if "innerdata_extrabkg_test.npy" in data_files:
            innerdata_test_path.append(os.path.join(args.data_dir, 'innerdata_extrabkg_test.npy'))
        extrasig_path = None
        if args.supervised:
            innerdata_train_path = []
            innerdata_val_path = []
            innerdata_train_path.append(os.path.join(args.data_dir, 'innerdata_extrasig_train.npy'))
            innerdata_val_path.append(os.path.join(args.data_dir, 'innerdata_extrasig_val.npy'))
            innerdata_train_path.append(os.path.join(args.data_dir, 'innerdata_extrabkg_train.npy'))
            innerdata_val_path.append(os.path.join(args.data_dir, 'innerdata_extrabkg_val.npy'))
            extra_bkg = None
        elif args.idealized_AD:
            extra_bkg = [os.path.join(args.data_dir, 'innerdata_extrabkg_train.npy'),
                         os.path.join(args.data_dir, 'innerdata_extrabkg_val.npy')]
        else:
            extra_bkg = None

    else:
        innerdata_train_path = os.path.join(args.data_dir, 'innerdata_train.npy')
        extrasig_path = os.path.join(args.data_dir, 'innerdata_extrasig.npy')
        if args.extra_bkg:
            extra_bkg = os.path.join(args.data_dir, 'innerdata_extrabkg.npy')
        else:
            extra_bkg = None
        innerdata_val_path = None
        innerdata_test_path = os.path.join(args.data_dir, 'innerdata_test.npy')

    # data preprocessing
    data = LHCORD_data_handler(innerdata_train_path,
                               innerdata_test_path,
                               os.path.join(args.data_dir, 'outerdata_train.npy'),
                               os.path.join(args.data_dir, 'outerdata_test.npy'),
                               extrasig_path,
                               inner_extrabkg_path=extra_bkg,
                               inner_val_path=innerdata_val_path,
                               batch_size=256,
                               device=device)
    if args.datashift != 0:
        print("applying a datashift of", args.datashift)
        data.shift_data(args.datashift, constant_shift=False, random_shift=args.random_shift,
                        shift_mj1=True, shift_dm=True, additional_shift=False)

    if args.CWoLa:
        # data preprocessing
        samples = None
        data.preprocess_CWoLa_data(fiducial_cut=args.fiducial_cut, no_logit=args.no_logit,
                                   outer_range=(args.SR_min-0.2, args.SR_max+0.2))

    else:
        # data preprocessing
        data.preprocess_ANODE_data(fiducial_cut=args.fiducial_cut, no_logit=args.no_logit_trained,
                                   no_mean_shift=args.no_logit_trained)

        # model instantiation
        if len(args.external_samples) > 0:
            model_list = None
            loaded_samples = [np.load(sample_path) for sample_path in args.external_samples]
            external_sample = np.concatenate(loaded_samples)
        else:
            model_list = []
            for model_path in args.ANODE_models:
                anode = DensityEstimator(args.config_file,
                                         eval_mode=True,
                                         load_path=model_path,
                                         device=device, verbose=args.verbose,
                                         bound=args.no_logit_trained)
                model_list.append(anode.model)
            external_sample = None

        # generate samples
        if not args.supervised and not args.idealized_AD:
            uniform_cond = not args.realistic_conditional
            samples = sample_handler(model_list, args.n_samples, data, cond_min=args.SR_min,
                                     cond_max=args.SR_max, uniform_cond=uniform_cond,
                                     external_sample=external_sample,
                                     device=device, no_logit=args.no_logit_trained,
                                     no_mean_shift=args.no_logit_trained,
                                     KDE_bandwidth=args.KDE_bandwidth)
        else:
            samples = None

        # redo data preprocessing if the classifier should not use logit but ANODE did
        data.preprocess_ANODE_data(fiducial_cut=args.fiducial_cut, no_logit=args.no_logit,
                                   no_mean_shift=args.no_logit_trained)

        # sample preprocessing
        if not args.supervised and not args.idealized_AD:
            samples.preprocess_samples(fiducial_cut=args.fiducial_cut, no_logit=args.no_logit,
                                       no_mean_shift=args.no_logit_trained)


    # sample mixing
    X_train, y_train, X_test, y_test, X_extrasig, y_extrasig = mix_data_samples(
        data, samples_handler=samples, oversampling=args.oversampling,
        savedir=args.savedir, CWoLa=args.CWoLa, supervised=args.supervised,
        idealized_AD=args.idealized_AD, separate_val_set=args.separate_val_set or finer_data_split)

    # sanity checks
    if not args.CWoLa and not args.supervised and not args.idealized_AD:
        samples.sanity_check(savefig=os.path.join(args.savedir, "sanity_check"), suppress_show=True)
        samples.sanity_check_after_cuts(savefig=os.path.join(args.savedir, "sanity_check_cuts"),
                                        suppress_show=True)

    if args.supervised or args.separate_val_set or finer_data_split:
        X_val = X_extrasig
        if args.supervised:
            y_train = X_train[:, -1]
            y_test = X_test[:, -1]
            y_val = X_val[:, -1]
        else:
            y_val = X_val[:, -2]
        plot_data_sample_comparison(X_val, y_val, title="validation set",
                                    savefig=os.path.join(args.savedir,
                                                         "data_sample_comparison_val"),
                                    suppress_show=True)

    plot_data_sample_comparison(X_train, y_train, title="training set",
                                savefig=os.path.join(args.savedir, "data_sample_comparison_train"),
                                suppress_show=True)
    plot_data_sample_comparison(X_test, y_test, title="test set",
                                savefig=os.path.join(args.savedir, "data_sample_comparison_test"),
                                suppress_show=True)

    print("number of training data =", X_train.shape[0])
    print("number of test data =", X_test.shape[0])
    if not args.no_extra_signal:
        if args.supervised or args.separate_val_set or finer_data_split:
            print("number of validation data =", X_val.shape[0])
        elif extrasig_path is not None:
            print("number of extra signal data =", X_extrasig.shape[0])


if __name__ == "__main__":
    args_extern = parser.parse_args()
    main(args_extern)
