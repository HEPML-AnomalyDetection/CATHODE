import os
import argparse
import torch
from data_handler import LHCORD_data_handler
from density_estimator import DensityEstimator
from ANODE_evaluation_utils import compute_ANODE_score

parser = argparse.ArgumentParser(
    description='Compute the classic ANODE anomaly score.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--inner_models', type=str, nargs="+", default="",
                    help=('Space-separated list of inner ANODE model paths to be used for '
                          '(ensemble) likelihood prediction.'))
parser.add_argument('--outer_models', type=str, nargs="+", default="",
                    help=('Space-separated list of outer ANODE model paths to be used for '
                          '(ensemble) likelihood prediction.'))
parser.add_argument('--config_file', type=str, default="ANODE_model.yml",
                    help='ANODE model config file (.yml).')
parser.add_argument('--savedir', type=str, default="ANODE_output/",
                    help='Path to directory where output files will be stored.')
parser.add_argument('--data_dir', type=str, default="./",
                    help='Path to the input data.')
parser.add_argument('--fiducial_cut', action="store_true", default=False,
                    help='Whether to apply an (ANODE paper) fiducial cut on the data.')
parser.add_argument('--datashift', type=float, default=0.,
                    help='Shift on jet mass variables to be applied.')
parser.add_argument('--random_shift', action="store_true", default=False,
                    help='Shift is not correlated to the actual mjj but randomized.')
parser.add_argument('--no_extra_signal', action="store_true", default=False,
                    help='Suppress the evaluation on extra signal.')

@torch.no_grad()
def main(args):

    # selecting appropriate device
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda:0" if CUDA else "cpu")

    # checking for data separation
    data_files = os.listdir(args.data_dir)
    if not args.no_extra_signal:
        if "innerdata_extrasig_test.npy" in data_files:
            extrasig_path = os.path.join(args.data_dir, 'innerdata_extrasig_test.npy')
        elif "innerdata_extrasig.npy" in data_files:
            extrasig_path = os.path.join(args.data_dir, 'innerdata_extrasig.npy')
    else:
        extrasig_path = None

    innerdata_test_path = [os.path.join(args.data_dir, 'innerdata_test.npy')]
    if "innerdata_extrabkg_test.npy" in data_files:
        innerdata_test_path.append(os.path.join(args.data_dir, 'innerdata_extrabkg_test.npy'))

    # data preprocessing
    data = LHCORD_data_handler(os.path.join(args.data_dir, 'innerdata_train.npy'),
                               innerdata_test_path,
                               os.path.join(args.data_dir, 'outerdata_train.npy'),
                               os.path.join(args.data_dir, 'outerdata_test.npy'),
                               extrasig_path,
                               batch_size=256,
                               device=device)
    if args.datashift != 0:
        print("applying a datashift of", args.datashift)
        data.shift_data(args.datashift, constant_shift=False, random_shift=args.random_shift,
                        shift_mj1=True, shift_dm=True, additional_shift=False)

    data.preprocess_ANODE_data(fiducial_cut=args.fiducial_cut)
    data.preprocess_classic_ANODE_data(fiducial_cut=args.fiducial_cut)

    init_inner_model_list = []
    for model_path in args.inner_models:
        init_inner_model_list.append(DensityEstimator(args.config_file, eval_mode=True,
                                                      load_path=model_path, device=device)[0])

    init_outer_model_list = []
    for model_path in args.outer_models:
        init_outer_model_list.append(DensityEstimator(args.config_file, eval_mode=True,
                                                      load_path=model_path, device=device)[0])

    preds, sig_labels, preds_extrasig = compute_ANODE_score(init_inner_model_list,
                                                            init_outer_model_list, data, device,
                                                            savedir=args.savedir,
                                                            extra_signal=not args.no_extra_signal)


if __name__ == "__main__":
    args_extern = parser.parse_args()
    main(args_extern)
