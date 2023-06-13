# pylint: disable=invalid-name
""" Runs the full pipline of CATHODE """

import os
import argparse
import numpy as np

from run_ANODE_training import main as train_DE
from run_classifier_data_creation import main as create_data
from run_classifier_training import main as train_classifier
from run_ANODE_evaluation import main as eval_ANODE
from evaluation_utils import full_single_evaluation, classic_ANODE_eval, minimum_val_loss_model_evaluation

parser = argparse.ArgumentParser(
    description='Run the full CATHODE analysis chain.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


# TODO: - collect and plot data of DE-LL vs max-SIC


# generic options
parser.add_argument('--mode', type=str, default='CATHODE', choices=["CATHODE", "ANODE", "CWoLa", "idealized_AD", "supervised"],
                    help=('Which mode to run the code in. Must be one of '
                          '["CATHODE", "ANODE", "CWoLa", "idealized_AD", "supervised"]'))

parser.add_argument('--data_dir', type=str, default="separated_data/",
                    help='Path to the input data.')
parser.add_argument('--save_dir', type=str, default="CATHODE_models/",
                    help='Path to directory where model files will be stored.')
parser.add_argument('--datashift', type=float, default=0.,
                    help='Shift on jet mass variables to be applied.')
parser.add_argument('--random_shift', action="store_true", default=False,
                    help='Shift is not correlated to the actual mjj but randomized.')
parser.add_argument('--fiducial_cut', action="store_true", default=False,
                    help='Whether to apply an (ANODE paper) fiducial cut on'
                    ' the data (and samples).')
parser.add_argument('--no_extra_signal', action="store_true", default=False,
                    help='Suppress the processing of the extra signal sample.')
parser.add_argument('--verbose', action="store_true", default=False,
                    help='Enables more printout.')

# Density Estimator specific options
parser.add_argument('--DE_config_file', type=str, default="DE_MAF_model.yml",
                    help='ANODE model config file (.yml).')
parser.add_argument('--DE_epochs', type=int, default=100,
                    help='Number of Density Estimation training epochs.')
parser.add_argument('--DE_batch_size', type=int, default=256,
                    help='Batch size during density estimation training.')
parser.add_argument('--DE_skip', action="store_true", default=False,
                    help='Skips the density estimation (loads existing files instead).')

parser.add_argument('--DE_no_logit', action="store_true", default=False,
                    help='Turns off the logit transform in the density estimator.')
parser.add_argument('--DE_file_name', type=str, default="my_ANODE_model",
                    help='File name for the density estimator.')

# Classifier specific options
parser.add_argument('--cf_config_file', type=str, default="classifier.yml",
                    help='Classifier model config file (.yml).')
parser.add_argument('--cf_epochs', type=int, default=100,
                    help='Number of classifier training epochs')
parser.add_argument('--cf_n_samples', type=int, default=130000,
                    help='Number of samples to be generated. Currently the samples will be '
                    'cut down to match data proportion.')
parser.add_argument('--cf_realistic_conditional', action="store_true", default=False,
                    help='Sample the conditional from a KDE fit rather than a '
                    'uniform distribution.')
parser.add_argument('--cf_KDE_bandwidth', type=float, default=0.01,
                    help='Bandwith of the KDE fit (used when realistic_conditional is selected)')
parser.add_argument('--cf_oversampling', action="store_true", default=False,
                    help='Add the full number of samples to the training set rather than mixing '
                    'it in equal parts with data.')
parser.add_argument('--cf_no_logit', action="store_true", default=False,
                    help='Turns off logit tranform in the classifier.')
parser.add_argument('--cf_external_samples', type=str, nargs="+", default="",
                    help='Space-separated list of pre-sampled npy files of physical variables if '
                    'the sampling has been done externally. The format is '
                    '(mjj, mj1, dmj, tau21_1, tau21_2)')
parser.add_argument('--cf_SR_min', type=float, default=3.3,
                    help='Lower boundary of signal region.')
parser.add_argument('--cf_SR_max', type=float, default=3.7,
                    help='Upper boundary of signal region.')
parser.add_argument('--cf_n_runs', type=int, default=10,
                    help='Number of independent classifier training runs.')
parser.add_argument('--cf_batch_size', type=int, default=128,
                    help='Batch size during classifier training.')
parser.add_argument('--cf_use_mjj', action="store_true", default=False,
                    help='Use the conditional variable as classifier input during training.')
parser.add_argument('--cf_use_class_weights', action="store_true", default=False,
                    help=('Weight the classes according to their occurence in the training set. '
                          'Necessary if the training set was intentionally oversampled.'))
parser.add_argument('--cf_SR_center', type=float, default=3.5,
                    help='Central value of signal region. Must only be given for using '
                    'CWoLa with weights.')
parser.add_argument('--cf_extra_bkg', action="store_true", default=False,
                    help='Make use of extra background (for supervised and idealized AD).')
parser.add_argument('--cf_separate_val_set', action="store_true", default=False,
                    help='Define a separate validation set to pick the classifier epochs.')
parser.add_argument('--cf_save_model', action="store_true", default=False,
                    help=('Save the tensorflow model after each epoch instead of '
                          'saving predictions.'))
parser.add_argument('--cf_skip_create', action="store_true", default=False,
                    help='Skips the creation of the classifier dataset '
                    '(loads existing files instead).')
parser.add_argument('--cf_skip_train', action="store_true", default=False,
                    help='Skips the training of the classifier '
                    '(loads existing files instead).')


def create_namespace_DE_training(arg):
    """ creates namespace for density estimation training script"""
    ret = argparse.Namespace()

    ret.config_file = arg.DE_config_file
    ret.data_dir = arg.data_dir
    ret.savedir = arg.save_dir
    ret.datashift = arg.datashift
    ret.random_shift = arg.random_shift
    ret.verbose = arg.verbose

    ret.model_file_name = arg.DE_file_name
    ret.epochs = arg.DE_epochs
    ret.batch_size = arg.DE_batch_size

    ret.no_logit = arg.DE_no_logit
    ret.inner_model = False
    return ret

def create_namespace_classifier_creation(arg):
    """ creates namespace for classifier dataset creation script """
    ret = argparse.Namespace()

    ret.savedir = arg.save_dir
    ret.datashift = arg.datashift
    ret.data_dir = arg.data_dir
    ret.random_shift = arg.random_shift
    ret.config_file = arg.DE_config_file
    ret.verbose = arg.verbose

    ret.fiducial_cut = arg.fiducial_cut
    ret.n_samples = arg.cf_n_samples
    ret.realistic_conditional = arg.cf_realistic_conditional
    ret.KDE_bandwidth = arg.cf_KDE_bandwidth
    ret.oversampling = arg.cf_oversampling
    ret.no_extra_signal = arg.no_extra_signal
    ret.CWoLa = False
    ret.supervised = False
    ret.idealized_AD = False
    ret.no_logit = arg.cf_no_logit
    ret.no_logit_trained = arg.DE_no_logit
    ret.external_samples = arg.cf_external_samples
    ret.SR_min = arg.cf_SR_min
    ret.SR_max = arg.cf_SR_max
    ret.extra_bkg = arg.cf_extra_bkg
    ret.separate_val_set = arg.cf_separate_val_set
    ret.ANODE_models = []
    return ret

def create_namespace_classifier_training(arg):
    """ creats namespace for classifier training script """
    ret = argparse.Namespace()
    ret.config_file = arg.cf_config_file
    ret.data_dir = arg.save_dir #classifier data was previously saved in --save_dir
    ret.savedir = arg.save_dir
    ret.verbose = arg.verbose

    ret.epochs = arg.cf_epochs
    ret.n_runs = arg.cf_n_runs
    ret.batch_size = arg.cf_batch_size
    ret.no_extra_signal = arg.no_extra_signal
    ret.use_mjj = arg.cf_use_mjj
    ret.supervised = False
    if arg.cf_oversampling:
        ret.use_class_weights = True
    else:
        ret.use_class_weights = arg.cf_use_class_weights
    ret.CWoLa = False
    ret.SR_center = arg.cf_SR_center
    ret.save_model = arg.cf_save_model
    ret.separate_val_set = arg.cf_separate_val_set
    return ret

def create_namespace_classic_ANODE(arg):
    """ creates namespace for classic ANODE evaluation """
    ret = argparse.Namespace()
    ret.savedir = arg.save_dir
    ret.datashift = arg.datashift
    ret.data_dir = arg.data_dir
    ret.random_shift = arg.random_shift
    ret.config_file = arg.DE_config_file
    ret.fiducial_cut = arg.fiducial_cut
    ret.no_extra_signal = arg.no_extra_signal

    return ret

def find_best_epochs(arg, num_models):
    """ looks through saved val-losses and creates list-of-paths of num_models best ones"""
    val_losses = np.load(os.path.join(arg.savedir, arg.model_file_name+"_val_losses.npy"))
    idx = np.argpartition(val_losses, num_models)[:num_models] #faster than argsort
    ret_list = []
    for index in idx:
        ret_list.append(os.path.join(arg.savedir,
                                     arg.model_file_name+'_epoch_'+str(index-1)+'.par'))
    return ret_list

if __name__ == '__main__':
    args = parser.parse_args()

    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print("running "+args.mode)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.mode == 'CATHODE':

        training_DE_args = create_namespace_DE_training(args)
        if not args.DE_skip:
            print("  * * *  train density estimator  * * *  ")
            train_DE(training_DE_args)

        if not args.cf_skip_create:
            print("  * * *  create classifier dataset  * * *  ")
            create_data_args = create_namespace_classifier_creation(args)
            create_data_args.ANODE_models = find_best_epochs(training_DE_args, 10)
            create_data(create_data_args)

        if not args.cf_skip_train:
            print("  * * *  train classifier  * * *  ")
            training_cf_args = create_namespace_classifier_training(args)
            train_classifier(training_cf_args)

        print("  * * *  evaluate classifier  * * *  ")
        _ = full_single_evaluation(args.save_dir, args.save_dir, n_ensemble_epochs=10,
                                   extra_signal=not args.no_extra_signal, sic_range=(0, 20),
                                   savefig=os.path.join(args.save_dir, 'result_SIC'))

    elif args.mode == 'ANODE':

        print("  * * *  train outer density estimator  * * *  ")
        training_DE_args = create_namespace_DE_training(args)
        if not args.DE_skip:
            train_DE(training_DE_args)
        outer_best = find_best_epochs(training_DE_args, 1)

        print("  * * *  train inner density estimator  * * *  ")
        training_DE_args.inner_model = True
        training_DE_args.model_file_name += '_inner'
        if not args.DE_skip:
            train_DE(training_DE_args)
        inner_best = find_best_epochs(training_DE_args, 10)

        print("  * * *  evaluate ANODE  * * *  ")
        eval_ANODE_args = create_namespace_classic_ANODE(args)
        eval_ANODE_args.outer_models = outer_best
        eval_ANODE_args.inner_models = inner_best
        eval_ANODE(eval_ANODE_args)
        _ = classic_ANODE_eval(args.save_dir, extra_signal=not args.no_extra_signal,
                               savefig=os.path.join(args.save_dir, 'result_SIC'))

    elif args.mode == 'CWoLa':
        if not args.cf_skip_create:
            print("  * * *  create classifier dataset  * * *  ")
            create_data_args = create_namespace_classifier_creation(args)
            create_data_args.CWoLa = True
            create_data(create_data_args)

        if not args.cf_skip_train:
            print("  * * *  train classifier  * * *  ")
            classifier_training_args = create_namespace_classifier_training(args)
            classifier_training_args.CWoLa = True
            train_classifier(classifier_training_args)

        print("  * * *  evaluate classifier  * * *  ")
        _ = full_single_evaluation(args.save_dir, args.save_dir, n_ensemble_epochs=10,
                                   extra_signal=not args.no_extra_signal, sic_range=(0, 20),
                                   savefig=os.path.join(args.save_dir, 'result_SIC'))

    elif args.mode == 'supervised':
        if not args.cf_skip_create:
            print("  * * *  create classifier dataset  * * *  ")
            create_data_args = create_namespace_classifier_creation(args)
            create_data_args.supervised = True
            create_data(create_data_args)

        if not args.cf_skip_train:
            print("  * * *  train classifier  * * *  ")
            classifier_training_args = create_namespace_classifier_training(args)
            classifier_training_args.supervised = True
            train_classifier(classifier_training_args)

        print("  * * *  evaluate classifier  * * *  ")
        _ = full_single_evaluation(args.save_dir, args.save_dir, n_ensemble_epochs=10,
                                   extra_signal=not args.no_extra_signal, sic_range=(0, 20),
                                   savefig=os.path.join(args.save_dir, 'result_SIC'))

    elif args.mode == 'idealized_AD':
        if not args.cf_skip_create:
            print("  * * *  create classifier dataset  * * *  ")
            create_data_args = create_namespace_classifier_creation(args)
            create_data_args.idealized_AD = True
            create_data(create_data_args)

        if not args.cf_skip_train:
            print("  * * *  train classifier  * * *  ")
            classifier_training_args = create_namespace_classifier_training(args)
            train_classifier(classifier_training_args)

        print("  * * *  evaluate classifier  * * *  ")
        _ = full_single_evaluation(args.save_dir, args.save_dir, n_ensemble_epochs=10,
                                   extra_signal=not args.no_extra_signal, sic_range=(0, 20),
                                   savefig=os.path.join(args.save_dir, 'result_SIC'))

    else:
        raise ValueError('Wrong --mode given: {}'.format(args.mode))
