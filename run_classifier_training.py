import argparse
import os
import numpy as np
from classifier_training_utils import train_n_models, plot_classifier_losses
from evaluation_utils import minimum_val_loss_model_evaluation
import matplotlib as mpl
mpl.use('Agg')

parser = argparse.ArgumentParser(
    description=('Train the classifier on SR data vs samples (or vs SB data for CWoLa). '
                 'The same classifier can also be used for a fully supervised training on '
                 'signal labels.'),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--n_runs', type=int, default=10,
                    help='Number of independent training runs.')
parser.add_argument('--config_file', type=str, default="classifier.yml",
                    help='Classifier model config file (.yml).')
parser.add_argument('--data_dir', type=str, default="classifier_data/",
                    help='Input data directory.')
parser.add_argument('--savedir', type=str, default="classifier_output/",
                    help='Path to directory where output files will be stored.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size during training.')
parser.add_argument('--no_extra_signal', action="store_true", default=False,
                    help='Suppress the evaluation on extra signal.')
parser.add_argument('--use_mjj', action="store_true", default=False,
                    help='Use the conditional variable as classifier input during training.')
parser.add_argument('--supervised', action="store_true", default=False,
                    help=('Do a fully supervised training, i.e. only on real data and using the '
                          'actual signal/background labels.'))
parser.add_argument('--use_class_weights', action="store_true", default=False,
                    help=('Weight the classes according to their occurence in the training set. '
                          'Necessary if the training set was intentionally oversampled.'))
parser.add_argument('--CWoLa', action="store_true", default=False,
                    help=('Use sideband reweighting specific to CWoLa (only effective if '
                          'use_class_weights is selected).'))
parser.add_argument('--SR_center', type=float, default=3.5,
                    help='Central value of signal region. Must only be given for using '
                    'CWoLa with weights.')
parser.add_argument('--save_model', action="store_true", default=False,
                    help=('Save the tensorflow model after each epoch instead of '
                          'saving predictions.'))
parser.add_argument('--separate_val_set', action="store_true", default=False,
                    help='Run on a separate validation set to pick the classifier epochs.')
parser.add_argument('--verbose', action="store_true", default=False,
                    help='Higher verbosity during training.')

def main(args):

    # loading the data
    # TODO get rid of the y's since the information is fully included in X
    X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))
    if args.no_extra_signal or args.supervised:
        X_extrasig = None
    else:
        X_extrasig = np.load(os.path.join(args.data_dir, 'X_extrasig.npy'))
    if args.supervised or args.separate_val_set:
        X_val = np.load(os.path.join(args.data_dir, 'X_validation.npy'))        
    else:
        X_val = None

    if args.save_model:
        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir)
        save_model = os.path.join(args.savedir, "model")
    else:
        save_model = None

    # actual training
    loss_matris, val_loss_matris = train_n_models(
        args.n_runs, args.config_file, args.epochs, X_train, y_train, X_test, y_test,
        X_extrasig=X_extrasig, X_val=X_val, use_mjj=args.use_mjj, batch_size=args.batch_size,
        supervised=args.supervised, use_class_weights=args.use_class_weights,
        CWoLa=args.CWoLa, SR_center=args.SR_center, verbose=args.verbose,
        savedir=args.savedir, save_model=save_model)

    if args.save_model:
        minimum_val_loss_model_evaluation(args.data_dir, args.savedir, n_epochs=10,
                                use_mjj=args.use_mjj, extra_signal=not args.no_extra_signal)

    for i in range(loss_matris.shape[0]):
        plot_classifier_losses(
            loss_matris[i], val_loss_matris[i],
            savefig=save_model+"_run"+str(i)+"_loss_plot",
            suppress_show=True
        )

if __name__ == "__main__":
    args_extern = parser.parse_args()
    main(args_extern)
