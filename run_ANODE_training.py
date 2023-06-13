import argparse
import os
import torch
import numpy as np
import pickle
from data_handler import LHCORD_data_handler
from ANODE_training_utils import train_ANODE, plot_ANODE_losses
from density_estimator import DensityEstimator

parser = argparse.ArgumentParser(
    description='Train an ANODE model.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size during training.')
parser.add_argument('--config_file', type=str, default="ANODE_model.yml",
                    help='ANODE model config file (.yml).')
parser.add_argument('--data_dir', type=str, default="./",
                    help='Path to the input data.')
parser.add_argument('--datashift', type=float, default=0.,
                    help='Shift on jet mass variables to be applied.')
parser.add_argument('--random_shift', action="store_true", default=False,
                    help='Shift is not correlated to the actual mjj but randomized.')
parser.add_argument('--inner_model', action="store_true", default=False,
                    help='Trains on signal region rather than sidebands. Used for classic ANODE.')
parser.add_argument('--verbose', action="store_true", default=False,
                    help='Enables more printout.')

parser.add_argument('--model_file_name', type=str, default="my_ANODE_model",
                    help='Some ANODE model file name for book keeping.')
parser.add_argument('--no_logit', action="store_true", default=False,
                    help='Turns off the logit transform.')
parser.add_argument('--savedir', type=str, default="ANODE_models/",
                    help='Path to directory where model files will be stored.')


def prepare_processing_dict(source):
    out = {}
    out['max'] = source['max']
    out['min'] = source['min']
    out['mean2'] = source['mean2']
    out['std2'] = source['std2']
    out['std2_logit_fix'] = source['std2_logit_fix']
    return out


def main(args):
    # for debugging:
    # torch.manual_seed(2104)
    # np.random.seed(2104)

    # selecting appropriate device
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda:0" if CUDA else "cpu")

    data = LHCORD_data_handler(os.path.join(args.data_dir, 'innerdata_train.npy'),
                               os.path.join(args.data_dir, 'innerdata_val.npy'),
                               os.path.join(args.data_dir, 'outerdata_train.npy'),
                               os.path.join(args.data_dir, 'outerdata_val.npy'),
                               None,
                               batch_size=args.batch_size,
                               device=device)
    if args.datashift != 0:
        print("applying a datashift of", args.datashift)
        data.shift_data(args.datashift, constant_shift=False, random_shift=args.random_shift,
                        shift_mj1=True, shift_dm=True, additional_shift=False)
    data.preprocess_ANODE_data(no_logit=args.no_logit,
                               no_mean_shift=args.no_logit)
    if args.inner_model:
        train_loader = data.inner_ANODE_datadict_train['loader']
        test_loader = data.inner_ANODE_datadict_test['loader']
        data_std = data.inner_ANODE_datadict_train['std2_logit_fix']
        train_data_processing = prepare_processing_dict(data.inner_ANODE_datadict_train)
    else:
        train_loader = data.outer_ANODE_datadict_train['loader']
        test_loader = data.outer_ANODE_datadict_test['loader']
        data_std = data.outer_ANODE_datadict_train['std2_logit_fix']
        train_data_processing = prepare_processing_dict(data.outer_ANODE_datadict_train)
    pickle.dump(train_data_processing, open(os.path.join(args.savedir, 'data_processing.p'), 'wb'))

    # actual training
    anode = DensityEstimator(args.config_file, device=device,
                             verbose=args.verbose, bound=args.no_logit)
    model, optimizer = anode.model, anode.optimizer

    train_ANODE(model, optimizer, train_loader, test_loader, args.model_file_name,
                args.epochs, savedir=args.savedir, device=device, verbose=args.verbose,
                no_logit=args.no_logit, data_std=data_std)

    # plot losses
    train_losses = np.load(os.path.join(args.savedir, args.model_file_name+"_train_losses.npy"))
    val_losses = np.load(os.path.join(args.savedir, args.model_file_name+"_val_losses.npy"))
    plot_ANODE_losses(train_losses, val_losses, yrange=None,
                      savefig=os.path.join(args.savedir, args.model_file_name+"_loss_plot"),
                      suppress_show=True)


if __name__ == "__main__":
    args_extern = parser.parse_args()
    main(args_extern)
