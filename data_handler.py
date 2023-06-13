import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neighbors import KernelDensity


class LHCORD_data_handler:
    def __init__(self, inner_train_path, inner_test_path, outer_train_path,
                 outer_test_path, inner_extrasig_path, inner_extrabkg_path=None,
                 inner_val_path=None, batch_size=256, test_batch_size=None,
                 device=torch.device("cpu")):

        # class attributes
        self.datashift = 0. # shift applied after instantiating the data
        self.batch_size = batch_size
        if test_batch_size is None:
            self.test_batch_size = 10* self.batch_size
        else:
            self.test_batch_size = test_batch_size

        self.device = device

        # Format:
        # mjj (TeV), mjmin (TeV), mjmax-mjmin (TeV), tau21(mjmin), tau21 (mjmax), sigorbg label

        # To be used by the classifier
        self.innerdata_train = load_files(inner_train_path)

        # To be used by the classifier
        self.innerdata_test = load_files(inner_test_path)

        # To be used by the density estimator
        self.outerdata_train = load_files(outer_train_path)

        # To be used by the denstity estimator
        self.outerdata_test = load_files(outer_test_path)

        # Separate validation data for the classifier
        if inner_val_path is not None:
            self.innerdata_val = load_files(inner_val_path)
        else:
            self.innerdata_val = None

        # Extra signal for SIC curves
        if inner_extrasig_path is not None:
            self.innerdata_extrasig = load_files(inner_extrasig_path)
        else:
            self.innerdata_extrasig = None

        # Extra background for idealized methods
        if inner_extrabkg_path is not None:
            self.innerdata_extrabkg = load_files(inner_extrabkg_path)
        else:
            self.innerdata_extrabkg = None


        # original data - keeps track of data before shift.
        # If no shift is applied, just references the data above
        self.original_innerdata_train = self.innerdata_train
        self.original_innerdata_test = self.innerdata_test
        self.original_outerdata_train = self.outerdata_train
        self.original_outerdata_test = self.outerdata_test
        self.original_innerdata_val = self.innerdata_val
        self.original_innerdata_extrasig = self.innerdata_extrasig
        self.original_innerdata_extrabkg = self.innerdata_extrabkg

        # to be filled by later functions:
        self.fiducial_cut = None
        self.outer_ANODE_datadict_train = None
        self.inner_ANODE_datadict_train = None
        self.outer_ANODE_datadict_test = None
        self.inner_ANODE_datadict_test = None
        self.inner_ANODE_datadict_val = None
        self.inner_ANODE_datadict_extrasig = None
        self.inner_ANODE_datadict_extrabkg = None
        self.outer_CWoLa_datadict_train = None
        self.inner_CWoLa_datadict_train = None
        self.outer_CWoLa_datadict_test = None
        self.inner_CWoLa_datadict_test = None
        self.inner_CWoLa_datadict_val = None
        self.inner_CWoLa_datadict_extrasig = None
        self.inner_classic_ANODE_innermodel_datadict_train = None     
        self.inner_classic_ANODE_innermodel_datadict_test = None
        self.inner_classic_ANODE_innermodel_datadict_extrasig = None
        self.inner_classic_ANODE_outermodel_datadict_train = None     
        self.inner_classic_ANODE_outermodel_datadict_test = None
        self.inner_classic_ANODE_outermodel_datadict_extrasig = None
        self.outer_ANODE_inner_preprocessing_datadict_train = None
        self.outer_ANODE_inner_preprocessing_datadict_test = None


    def shift_data(self, strength, constant_shift=False, random_shift=False,
                   shift_mj1=True, shift_dm=True, additional_shift=False):
        # method to be used explicitly by the user to apply a datashift to do all the
        # book keeping under the hood
        # thought about putting this in the __init__() but that would significantly increase
        # the number of arguments
        ## TODO add some more assert statements for stability
        ## TODO add some explanation on what the different types of shifts do
        assert self.datashift == 0, "Data can only be shifted once!"

        if strength == 0:
            print("A shift of 0 results in no action...")
            return

        self.datashift = strength

        # store actual copy as original data
        self.original_innerdata_train = self.innerdata_train.copy()
        self.original_innerdata_test = self.innerdata_test.copy()
        self.original_outerdata_train = self.outerdata_train.copy()
        self.original_outerdata_test = self.outerdata_test.copy()
        if self.innerdata_val is not None:
            self.original_innerdata_val = self.innerdata_val.copy()           
        if self.innerdata_extrasig is not None: 
            self.original_innerdata_extrasig = self.innerdata_extrasig.copy()
        if self.innerdata_extrabkg is not None: 
            self.original_innerdata_extrabkg = self.innerdata_extrabkg.copy()

        # generate random mjj vectors for random smearing (if selected)
        if random_shift:
            random_mjj_train = np.concatenate((self.innerdata_train[:, 0],
                                               self.outerdata_train[:, 0]))
            random_mjj_test = np.concatenate((self.innerdata_test[:, 0],
                                              self.outerdata_test[:, 0]))
            if self.innerdata_val is not None:
                random_mjj_test = np.concatenate((self.innerdata_val[:, 0],
                                                  random_mjj_test))
               
            np.random.shuffle(random_mjj_train)
            np.random.shuffle(random_mjj_test)
        else:
            random_mjj_train = None
            random_mjj_test = None

        # actually apply shifts
        self.innerdata_train = apply_shift_to_dataset(
            self.innerdata_train, strength, constant_shift=constant_shift,
            random_shift=random_shift, shift_mj1=shift_mj1, shift_dm=shift_dm,
            mjj_vals=random_mjj_train, additional_shift=additional_shift
            )
        self.innerdata_test = apply_shift_to_dataset(
            self.innerdata_test, strength, constant_shift=constant_shift,
            random_shift=random_shift, shift_mj1=shift_mj1, shift_dm=shift_dm,
            mjj_vals=random_mjj_test, additional_shift=additional_shift
            )
        self.outerdata_train = apply_shift_to_dataset(
            self.outerdata_train, strength, constant_shift=constant_shift,
            random_shift=random_shift, shift_mj1=shift_mj1, shift_dm=shift_dm,
            mjj_vals=random_mjj_train, additional_shift=additional_shift
            )
        self.outerdata_test = apply_shift_to_dataset(
            self.outerdata_test, strength, constant_shift=constant_shift,
            random_shift=random_shift, shift_mj1=shift_mj1, shift_dm=shift_dm,
            mjj_vals=random_mjj_test, additional_shift=additional_shift
            )
        if self.innerdata_val is not None:
            if random_mjj_test is not None:
                mjj_vals = random_mjj_test[::-1]
            else:
                mjj_vals = None
            self.innerdata_val = apply_shift_to_dataset(
                self.innerdata_val, strength, constant_shift=constant_shift,
                random_shift=random_shift, shift_mj1=shift_mj1, shift_dm=shift_dm,
                mjj_vals=mjj_vals, additional_shift=additional_shift
                )
        if self.innerdata_extrasig is not None: 
            self.innerdata_extrasig = apply_shift_to_dataset(
                self.innerdata_extrasig, strength, constant_shift=constant_shift,
                random_shift=random_shift, shift_mj1=shift_mj1, shift_dm=shift_dm,
                mjj_vals=random_mjj_test, additional_shift=additional_shift
                )
        if self.innerdata_extrabkg is not None:
            self.innerdata_extrabkg = apply_shift_to_dataset(
                self.innerdata_extrabkg, strength, constant_shift=constant_shift,
                random_shift=random_shift, shift_mj1=shift_mj1, shift_dm=shift_dm,
                mjj_vals=random_mjj_test, additional_shift=additional_shift
                )


    def preprocess_ANODE_data(self, fiducial_cut=False, no_logit=False, no_mean_shift=False,
                              external_param=None):
        ## Does all the preprocessing for the ANODE data, including saving the min/max/mean/std.
        ##   One can turn off the logit transform via the no_logit option.
        ##   One can turn off the scaling to mean 0 and std 1 via the no_mean_shift option

        self.fiducial_cut = fiducial_cut

        self.outer_ANODE_datadict_train = load_dataset(
            self.outerdata_train,
            batch_size=self.batch_size,
            fiducial_cut=fiducial_cut,
            shuffle_loader=True,
            no_logit=no_logit, device=self.device,
            no_mean_shift=no_mean_shift,
            external_datadict=external_param)
        self.outer_ANODE_datadict_test = load_dataset(
            self.outerdata_test,
            batch_size=self.test_batch_size,
            external_datadict=self.outer_ANODE_datadict_train if external_param is None \
            else external_param,
            fiducial_cut=fiducial_cut,
            shuffle_loader=False,
            no_logit=no_logit, device=self.device,
            no_mean_shift=no_mean_shift)
        self.inner_ANODE_datadict_train = load_dataset(
            self.innerdata_train,
            batch_size=self.batch_size,
            fiducial_cut=fiducial_cut,
            shuffle_loader=True,
            no_logit=no_logit, device=self.device,
            no_mean_shift=no_mean_shift,
            external_datadict=external_param)
        self.inner_ANODE_datadict_test = load_dataset(
            self.innerdata_test,
            batch_size=self.test_batch_size,
            external_datadict=self.inner_ANODE_datadict_train if external_param is None \
            else external_param,
            fiducial_cut=fiducial_cut,
            shuffle_loader=False,
            no_logit=no_logit, device=self.device,
            no_mean_shift=no_mean_shift)
        if self.innerdata_val is not None:
            self.inner_ANODE_datadict_val = load_dataset(
                self.innerdata_val,
                batch_size=self.test_batch_size,
                external_datadict=self.inner_ANODE_datadict_train if external_param is None \
                else external_param,
                fiducial_cut=fiducial_cut,
                shuffle_loader=False,
                no_logit=no_logit, device=self.device,
                no_mean_shift=no_mean_shift)
        if self.innerdata_extrasig is not None:
            self.inner_ANODE_datadict_extrasig = load_dataset(
                self.innerdata_extrasig,
                batch_size=self.test_batch_size,
                external_datadict=self.inner_ANODE_datadict_train if external_param is None \
                else external_param,
                fiducial_cut=fiducial_cut,
                shuffle_loader=False,
                no_logit=no_logit, device=self.device,
                no_mean_shift=no_mean_shift)
        if self.innerdata_extrabkg is not None:
            self.inner_ANODE_datadict_extrabkg = load_dataset(
                self.innerdata_extrabkg,
                batch_size=self.test_batch_size,
                external_datadict=self.inner_ANODE_datadict_train if external_param is None \
                else external_param,
                fiducial_cut=fiducial_cut,
                shuffle_loader=False,
                no_logit=no_logit, device=self.device,
                no_mean_shift=no_mean_shift)


    def preprocess_ANODE_outer_data(self, no_logit=False, no_mean_shift=False):
        ## Does the preprocessing for the ANODE data on the sideband, using consistent
        ##   standardization with the signal region. Can be used to evaluate ANODE/CATHODE
        ##   on the full spectrum. Needs to be done after preprocess_ANODE_data()
        ##   One can turn off the logit transform via the no_logit option.
        ##   One can turn off the scaling to mean 0 and std 1 via the no_mean_shift option

        self.outer_ANODE_inner_preprocessing_datadict_train = load_dataset(
            self.outerdata_train,
            batch_size=self.batch_size,
            fiducial_cut=self.fiducial_cut,
            external_datadict=self.inner_ANODE_datadict_train,
            shuffle_loader=True,
            no_logit=no_logit, device=self.device,
            no_mean_shift=no_mean_shift)
        self.outer_ANODE_inner_preprocessing_datadict_test = load_dataset(
            self.outerdata_test,
            batch_size=10*self.batch_size,
            external_datadict=self.inner_ANODE_datadict_train,
            fiducial_cut=self.fiducial_cut,
            shuffle_loader=False,
            no_logit=no_logit, device=self.device,
            no_mean_shift=no_mean_shift)
       

    def preprocess_CWoLa_data(self, fiducial_cut=False, no_logit=False, outer_range=(3.1, 3.9)):
        ## Basically the same function as above but normalizes inner data exactly like outer data.
        ##   One key difference is that only the adjacent 200 GeV strips of the two sidebands
        ##   are used.

        self.outer_CWoLa_datadict_train = load_dataset(
            self.outerdata_train,
            batch_size=self.batch_size,
            fiducial_cut=fiducial_cut,
            cond_range=outer_range,
            no_logit=no_logit, device=self.device)
        self.outer_CWoLa_datadict_test = load_dataset(
            self.outerdata_test,
            batch_size=self.test_batch_size,
            external_datadict=self.outer_CWoLa_datadict_train,
            fiducial_cut=fiducial_cut, cond_range=outer_range,
            no_logit=no_logit, device=self.device)
        self.inner_CWoLa_datadict_train = load_dataset(
            self.innerdata_train,
            batch_size=self.batch_size,
            external_datadict=self.outer_CWoLa_datadict_train,
            fiducial_cut=fiducial_cut, no_logit=no_logit,
            device=self.device)
        self.inner_CWoLa_datadict_test = load_dataset(
            self.innerdata_test,
            batch_size=self.test_batch_size,
            external_datadict=self.outer_CWoLa_datadict_train,
            fiducial_cut=fiducial_cut, no_logit=no_logit,
            device=self.device)
        if self.innerdata_val is not None:
            self.inner_CWoLa_datadict_val = load_dataset(
                self.innerdata_val,
                batch_size=self.test_batch_size,
                external_datadict=self.outer_CWoLa_datadict_train,
                fiducial_cut=fiducial_cut, no_logit=no_logit,
                device=self.device)
        if self.innerdata_extrasig is not None:
            self.inner_CWoLa_datadict_extrasig = load_dataset(
                self.innerdata_extrasig,
                batch_size=self.test_batch_size,
                external_datadict=self.outer_CWoLa_datadict_train,
                fiducial_cut=fiducial_cut, no_logit=no_logit,
                device=self.device)

    def preprocess_classic_ANODE_data(self, fiducial_cut=False):
        # Basically the same function as above but normalizes inner data for the outer model
        #   (for classic ANODE).
        #   An important aspect of this function is that it first derives the masking by both
        #   outer and innner model normalizations, so only data that survive both are taken into
        #   account.

        assert self.outer_ANODE_datadict_train is not None, (
            "Need to do call preprocess_ANODE_data() first!")

        train_mask_inner = load_dataset(self.innerdata_train,
                                        batch_size=self.batch_size,
                                        external_datadict=self.inner_ANODE_datadict_train,
                                        fiducial_cut=fiducial_cut, shuffle_loader=False,
                                        device=self.device)['mask'].detach().cpu().numpy()
        train_mask_outer = load_dataset(self.innerdata_train,
                                        batch_size=self.batch_size,
                                        external_datadict=self.outer_ANODE_datadict_train,
                                        fiducial_cut=fiducial_cut, shuffle_loader=False,
                                        device=self.device)['mask'].detach().cpu().numpy()
        test_mask_inner = load_dataset(self.innerdata_test,
                                       batch_size=self.test_batch_size,
                                       external_datadict=self.inner_ANODE_datadict_test,
                                       fiducial_cut=fiducial_cut, shuffle_loader=False,
                                       device=self.device)['mask'].detach().cpu().numpy()
        test_mask_outer = load_dataset(self.innerdata_test,
                                       batch_size=self.test_batch_size,
                                       external_datadict=self.outer_ANODE_datadict_test,
                                       fiducial_cut=fiducial_cut, shuffle_loader=False,
                                       device=self.device)['mask'].detach().cpu().numpy()
        if self.innerdata_extrasig is not None:
            extrasig_mask_inner = load_dataset(self.innerdata_extrasig,
                                           batch_size=self.test_batch_size,
                                           external_datadict=self.inner_ANODE_datadict_test,
                                           fiducial_cut=fiducial_cut, shuffle_loader=False,
                                           device=self.device)['mask'].detach().cpu().numpy()
            extrasig_mask_outer = load_dataset(self.innerdata_extrasig,
                                           batch_size=self.test_batch_size,
                                           external_datadict=self.outer_ANODE_datadict_test,
                                           fiducial_cut=fiducial_cut, shuffle_loader=False,
                                           device=self.device)['mask'].detach().cpu().numpy()

        innerdata_train_masked = self.innerdata_train[
            np.logical_and(train_mask_inner, train_mask_outer)]
        innerdata_test_masked = self.innerdata_test[
            np.logical_and(test_mask_inner, test_mask_outer)]
        if self.innerdata_extrasig is not None:
            innerdata_extrasig_masked = self.innerdata_extrasig[
                np.logical_and(extrasig_mask_inner, extrasig_mask_outer)]

        self.inner_classic_ANODE_innermodel_datadict_train = load_dataset(
            innerdata_train_masked,
            batch_size=self.batch_size,
            external_datadict=self.inner_ANODE_datadict_train,
            fiducial_cut=fiducial_cut, shuffle_loader=False,
            device=self.device)
        self.inner_classic_ANODE_innermodel_datadict_test = load_dataset(
            innerdata_test_masked,
            batch_size=self.test_batch_size,
            external_datadict=self.inner_ANODE_datadict_train,
            fiducial_cut=fiducial_cut, shuffle_loader=False,
            device=self.device)
        self.inner_classic_ANODE_outermodel_datadict_train = load_dataset(
            innerdata_train_masked,
            batch_size=self.batch_size,
            external_datadict=self.outer_ANODE_datadict_train,
            fiducial_cut=fiducial_cut, shuffle_loader=False,
            device=self.device)
        self.inner_classic_ANODE_outermodel_datadict_test = load_dataset(
            innerdata_test_masked,
            batch_size=self.test_batch_size,
            external_datadict=self.outer_ANODE_datadict_train,
            fiducial_cut=fiducial_cut, shuffle_loader=False,
            device=self.device)
        if self.innerdata_extrasig is not None:
            self.inner_classic_ANODE_innermodel_datadict_extrasig = load_dataset(
                innerdata_extrasig_masked,
                batch_size=self.test_batch_size,
                external_datadict=self.inner_ANODE_datadict_train,
                fiducial_cut=fiducial_cut, shuffle_loader=False,
                device=self.device)
            self.inner_classic_ANODE_outermodel_datadict_extrasig = load_dataset(
                innerdata_extrasig_masked,
                batch_size=self.test_batch_size,
                external_datadict=self.outer_ANODE_datadict_train,
                fiducial_cut=fiducial_cut, shuffle_loader=False,
                device=self.device)


    def plot_data(self):
        # inspect data visually. Plots original and shifted data separately if a shift was applied
        shifted = self.datashift != 0
        mjj_range = (1, 10)
        mj1_range = (0, 2)
        dmj_range = (0, 2)
        tau21_j1_range = (0, 1)
        tau21_j2_range = (0, 1)
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        nbins = 50

        plt.subplot(5, 2, 1)
        plt.hist(self.outerdata_train[self.outerdata_train[:, -1] == 0][:, 0], range=mjj_range,
                 bins=nbins, alpha=0.7, label='mjj outer background', zorder=1, color=colors[0])
        plt.hist(self.outerdata_train[self.outerdata_train[:, -1] == 1][:, 0], range=mjj_range,
                 bins=nbins, alpha=0.7, label='mjj outer signal', zorder=2, color=colors[1])
        plt.hist(self.innerdata_train[self.innerdata_train[:, -1] == 0][:, 0], range=mjj_range,
                 bins=nbins, alpha=0.7, label='mjj inner background', zorder=3, color=colors[2])
        plt.hist(self.innerdata_train[self.innerdata_train[:, -1] == 1][:, 0], range=mjj_range,
                 bins=nbins, alpha=0.7, label='mjj inner signal', zorder=4, color=colors[3])
        plt.legend(loc='upper right')
        plt.yscale('log')
        plt.title('Training data')
        plt.xlabel('mjj [TeV]')

        plt.subplot(5, 2, 2)
        plt.hist(self.outerdata_test[self.outerdata_test[:, -1] == 0][:, 0], range=mjj_range,
                 bins=nbins, alpha=0.7, label='mjj outer background', zorder=1, color=colors[0])
        plt.hist(self.outerdata_test[self.outerdata_test[:, -1] == 1][:, 0], range=mjj_range,
                 bins=nbins, alpha=0.7, label='mjj outer signal', zorder=2, color=colors[1])
        plt.hist(self.innerdata_test[self.innerdata_test[:, -1] == 0][:, 0], range=mjj_range,
                 bins=nbins, alpha=0.7, label='mjj inner background', zorder=3, color=colors[2])
        plt.hist(self.innerdata_test[self.innerdata_test[:, -1] == 1][:, 0], range=mjj_range,
                 bins=nbins, alpha=0.7, label='mjj inner signal', zorder=4, color=colors[3])
        plt.legend(loc = 'upper right')
        plt.yscale('log')
        plt.title('Test data')
        plt.xlabel('mjj [TeV]')

        plt.subplot(5, 2, 3)
        plt.hist(self.outerdata_train[self.outerdata_train[:, -1] == 0][:, 1], range=mj1_range,
                 bins=nbins, alpha=0.7, label='mj1 outer background', zorder=1, color=colors[0])
        plt.hist(self.outerdata_train[self.outerdata_train[:, -1] == 1][:, 1], range=mj1_range,
                 bins=nbins, alpha=0.7, label='mj1 outer signal', zorder=2, color=colors[1])
        plt.hist(self.innerdata_train[self.innerdata_train[:, -1] == 0][:, 1], range=mj1_range,
                 bins=nbins, alpha=0.7, label='mj1 inner background', zorder=3, color=colors[2])
        plt.hist(self.innerdata_train[self.innerdata_train[:, -1] == 1][:, 1], range=mj1_range,
                 bins=nbins, alpha=0.7, label='mj1 inner signal', zorder=4, color=colors[3])
        if shifted:
            plt.hist(self.original_outerdata_train[self.original_outerdata_train[:, -1] == 0][:, 1],
                     range=mj1_range, bins=nbins, zorder=5, histtype='step', color=colors[0])
            plt.hist(self.original_outerdata_train[self.original_outerdata_train[:, -1] == 1][:, 1],
                     range=mj1_range, bins=nbins, zorder=6, histtype='step', color=colors[1])
            plt.hist(self.original_innerdata_train[self.original_innerdata_train[:, -1] == 0][:, 1],
                     range=mj1_range, bins=nbins, zorder=7, histtype='step', color=colors[2])
            plt.hist(self.original_innerdata_train[self.original_innerdata_train[:, -1] == 1][:, 1],
                     range=mj1_range, bins=nbins, zorder=8, histtype='step', color=colors[3])
            plt.plot(np.nan, np.nan, color="black", label="unshifted data")
        plt.legend(loc='upper right')
        plt.yscale('log')
        plt.title('Training data')
        plt.xlabel('mj1 [TeV]')

        plt.subplot(5, 2, 4)
        plt.hist(self.outerdata_test[self.outerdata_test[:, -1] == 0][:, 1], range=mj1_range,
                 bins=nbins, alpha=0.7, label='mj1 outer background', zorder=1, color=colors[0])
        plt.hist(self.outerdata_test[self.outerdata_test[:, -1] == 1][:, 1], range=mj1_range,
                 bins=nbins, alpha=0.7, label='mj1 outer signal', zorder=2, color=colors[1])
        plt.hist(self.innerdata_test[self.innerdata_test[:, -1] == 0][:, 1], range=mj1_range,
                 bins=nbins, alpha=0.7, label='mj1 inner background', zorder=3, color=colors[2])
        plt.hist(self.innerdata_test[self.innerdata_test[:, -1] == 1][:, 1], range=mj1_range,
                 bins=nbins, alpha=0.7, label='mj1 inner signal', zorder=4, color=colors[3])
        if shifted:
            plt.hist(self.original_outerdata_test[self.original_outerdata_test[:, -1] == 0][:, 1],
                     range=mj1_range, bins=nbins, zorder=5, histtype='step', color=colors[0])
            plt.hist(self.original_outerdata_test[self.original_outerdata_test[:, -1] == 1][:, 1],
                     range=mj1_range, bins=nbins, zorder=6, histtype='step', color=colors[1])
            plt.hist(self.original_innerdata_test[self.original_innerdata_test[:, -1] == 0][:, 1],
                     range=mj1_range, bins=nbins, zorder=7, histtype='step', color=colors[2])
            plt.hist(self.original_innerdata_test[self.original_innerdata_test[:, -1] == 1][:, 1],
                     range=mj1_range, bins=nbins, zorder=8, histtype='step', color=colors[3])
            plt.plot(np.nan, np.nan, color="black", label="unshifted data")
        plt.legend(loc='upper right')
        plt.yscale('log')
        plt.title('Test data')
        plt.xlabel('mj1 [TeV]')

        plt.subplot(5, 2, 5)
        plt.hist(self.outerdata_train[self.outerdata_train[:, -1] == 0][:, 2], range=dmj_range,
                 bins=nbins, alpha=0.7, label='dmj outer background', zorder=1, color=colors[0])
        plt.hist(self.outerdata_train[self.outerdata_train[:, -1] == 1][:, 2], range=dmj_range,
                 bins=nbins, alpha=0.7, label='dmj outer signal', zorder=2, color=colors[1])
        plt.hist(self.innerdata_train[self.innerdata_train[:, -1] == 0][:, 2], range=dmj_range,
                 bins=nbins, alpha=0.7, label='dmj inner background', zorder=3, color=colors[2])
        plt.hist(self.innerdata_train[self.innerdata_train[:, -1] == 1][:, 2], range=dmj_range,
                 bins=nbins, alpha=0.7, label='dmj inner signal', zorder=4, color=colors[3])
        if shifted:
            plt.hist(self.original_outerdata_train[self.original_outerdata_train[:, -1] == 0][:, 2],
                     range=dmj_range, bins=nbins, zorder=5, histtype='step', color=colors[0])
            plt.hist(self.original_outerdata_train[self.original_outerdata_train[:, -1] == 1][:, 2],
                     range=dmj_range, bins=nbins, zorder=6, histtype='step', color=colors[1])
            plt.hist(self.original_innerdata_train[self.original_innerdata_train[:, -1] == 0][:, 2],
                     range=dmj_range, bins=nbins, zorder=7, histtype='step', color=colors[2])
            plt.hist(self.original_innerdata_train[self.original_innerdata_train[:, -1] == 1][:, 2],
                     range=dmj_range, bins=nbins, zorder=8, histtype='step', color=colors[3])
            plt.plot(np.nan, np.nan, color="black", label="unshifted data")
        plt.legend(loc='upper right')
        plt.yscale('log')
        plt.title('Training data')
        plt.xlabel('dmj [TeV]')

        plt.subplot(5, 2, 6)
        plt.hist(self.outerdata_test[self.outerdata_test[:, -1] == 0][:, 2], range=dmj_range,
                 bins=nbins, alpha=0.7, label='dmj outer background', zorder=1, color=colors[0])
        plt.hist(self.outerdata_test[self.outerdata_test[:, -1] == 1][:, 2], range=dmj_range,
                 bins=nbins, alpha=0.7, label='dmj outer signal', zorder=2, color=colors[1])
        plt.hist(self.innerdata_test[self.innerdata_test[:, -1] == 0][:, 2], range=dmj_range,
                 bins=nbins, alpha=0.7, label='dmj inner background', zorder=3, color=colors[2])
        plt.hist(self.innerdata_test[self.innerdata_test[:, -1] == 1][:, 2], range=dmj_range,
                 bins=nbins, alpha=0.7, label='dmj inner signal', zorder=4, color=colors[3])
        if shifted:
            plt.hist(self.original_outerdata_test[self.original_outerdata_test[:, -1] == 0][:, 2],
                     range=dmj_range, bins=nbins, zorder=5, histtype='step', color=colors[0])
            plt.hist(self.original_outerdata_test[self.original_outerdata_test[:, -1] == 1][:, 2],
                     range=dmj_range, bins=nbins, zorder=6, histtype='step', color=colors[1])
            plt.hist(self.original_innerdata_test[self.original_innerdata_test[:, -1] == 0][:, 2],
                     range=dmj_range, bins=nbins, zorder=7, histtype='step', color=colors[2])
            plt.hist(self.original_innerdata_test[self.original_innerdata_test[:, -1] == 1][:, 2],
                     range=dmj_range, bins=nbins, zorder=8, histtype='step', color=colors[3])
            plt.plot(np.nan, np.nan, color="black", label="unshifted data")
        plt.legend(loc='upper right')
        plt.yscale('log')
        plt.title('Test data')
        plt.xlabel('dmj [TeV]')

        plt.subplot(5, 2, 7)
        plt.hist(self.outerdata_train[self.outerdata_train[:, -1] == 0][:, 3], range=tau21_j1_range,
                 bins=nbins, alpha=0.7, label='tau21_j1 outer background', zorder=1,
                 color=colors[0])
        plt.hist(self.outerdata_train[self.outerdata_train[:, -1] == 1][:, 3], range=tau21_j1_range,
                 bins=nbins, alpha=0.7, label='tau21_j1 outer signal', zorder=2, color=colors[1])
        plt.hist(self.innerdata_train[self.innerdata_train[:, -1] == 0][:, 3], range=tau21_j1_range,
                 bins=nbins, alpha=0.7, label='tau21_j1 inner background', zorder=3,
                 color=colors[2])
        plt.hist(self.innerdata_train[self.innerdata_train[:, -1] == 1][:, 3], range=tau21_j1_range,
                 bins=nbins, alpha=0.7, label='tau21_j1 inner signal', zorder=4, color=colors[3])
        plt.legend(loc='upper right')
        plt.yscale('log')
        plt.title('Training data')
        plt.xlabel('tau21_j1')

        plt.subplot(5, 2, 8)
        plt.hist(self.outerdata_test[self.outerdata_test[:, -1] == 0][:, 3], range=tau21_j1_range,
                 bins=nbins, alpha=0.7, label='tau21_j1 outer background', zorder=1,
                 color=colors[0])
        plt.hist(self.outerdata_test[self.outerdata_test[:, -1] == 1][:, 3], range=tau21_j1_range,
                 bins=nbins, alpha=0.7, label='tau21_j1 outer signal', zorder=2, color=colors[1])
        plt.hist(self.innerdata_test[self.innerdata_test[:, -1] == 0][:, 3], range=tau21_j1_range,
                 bins=nbins, alpha=0.7, label='tau21_j1 inner background', zorder=3,
                 color=colors[2])
        plt.hist(self.innerdata_test[self.innerdata_test[:, -1] == 1][:, 3], range=tau21_j1_range,
                 bins=nbins, alpha=0.7, label='tau21_j1 inner signal', zorder=4, color=colors[3])
        plt.legend(loc='upper right')
        plt.yscale('log')
        plt.title('Test data')
        plt.xlabel('tau21_j1')

        plt.subplot(5, 2, 9)
        plt.hist(self.outerdata_train[self.outerdata_train[:, -1] == 0][:, 4], range=tau21_j2_range,
                 bins=nbins, alpha=0.7, label='tau21_j2 outer background', zorder=1,
                 color=colors[0])
        plt.hist(self.outerdata_train[self.outerdata_train[:, -1] == 1][:, 4], range=tau21_j2_range,
                 bins=nbins, alpha=0.7, label='tau21_j2 outer signal', zorder=2, color=colors[1])
        plt.hist(self.innerdata_train[self.innerdata_train[:, -1] == 0][:, 4], range=tau21_j2_range,
                 bins=nbins, alpha=0.7, label='tau21_j2 inner background', zorder=3,
                 color=colors[2])
        plt.hist(self.innerdata_train[self.innerdata_train[:, -1] == 1][:, 4], range=tau21_j2_range,
                 bins=nbins, alpha=0.7, label='tau21_j2 inner signal', zorder=4, color=colors[3])
        plt.legend(loc='upper right')
        plt.yscale('log')
        plt.title('Training data')
        plt.xlabel('tau21_j2')

        plt.subplot(5, 2, 10)
        plt.hist(self.outerdata_test[self.outerdata_test[:, -1] == 0][:, 4], range=tau21_j2_range,
                 bins=nbins, alpha=0.7, label='tau21_j2 outer background', zorder=1,
                 color=colors[0])
        plt.hist(self.outerdata_test[self.outerdata_test[:, -1] == 1][:, 4], range=tau21_j2_range,
                 bins=nbins, alpha=0.7, label='tau21_j2 outer signal', zorder=2, color=colors[1])
        plt.hist(self.innerdata_test[self.innerdata_test[:, -1] == 0][:, 4], range=tau21_j2_range,
                 bins=nbins, alpha=0.7, label='tau21_j2 inner background', zorder=3,
                 color=colors[2])
        plt.hist(self.innerdata_test[self.innerdata_test[:, -1] == 1][:, 4], range=tau21_j2_range,
                 bins=nbins, alpha=0.7, label='tau21_j2 inner signal', zorder=4, color=colors[3])
        plt.legend(loc='upper right')
        plt.yscale('log')
        plt.title('Test data')
        plt.xlabel('tau21_j2')

        plt.subplots_adjust(right=2.0, top=5.0)
        plt.show()
        plt.close()


class sample_handler:

    @torch.no_grad()
    def __init__(self, model_list, n_samples, data_handler, cond_min=3.3, cond_max=3.7,
                 uniform_cond=True, external_sample=None, device=torch.device("cpu"),
                 no_logit=False, no_mean_shift=False, KDE_bandwidth=0.01):

        if external_sample is None:
            assert data_handler.outer_ANODE_datadict_train is not None, (
                "Input data_handler needs to be preprocessed!")

        # simple class attributes
        self.model_list = model_list
        self.n_samples = n_samples
        self.data_handler = data_handler
        self.cond_min = cond_min
        self.cond_max = cond_max
        self.uniform_cond = uniform_cond
        self.device = device
        self.no_logit = no_logit
        self.no_mean_shift = no_mean_shift
        self.KDE_bandwidth = KDE_bandwidth ## used when the conditional is not sampled uniformly

        # actual samples
        if external_sample is not None:
            assert model_list is None, (
                "No model list should be provided if external samples are used!")
            self.sample_array = external_sample[:n_samples, 1:]
            self.cond_array = external_sample[:n_samples, 0]
        else:
            self.sample_array, self.cond_array = self.produce_samples()

        # keeping original data if shift is applied. First only a pointer to same object.
        self.original_sample_array = self.sample_array

        # to be filled by later functions
        self.sample_shift = 0
        self.preprocessed_samples = None
        self.preprocessed_cond_array = None
        self.masked_raw_samples = None # to store physical sample with domain/fiducial cuts applied
        self.masked_raw_cond_array = None
        self.fiducial_cut = None


    @torch.no_grad()
    def produce_samples(self):
        # sample mjj
        if self.uniform_cond: # sample mjj uniformly
            uni_mjj = np.random.uniform(low=self.cond_min, high=self.cond_max,
                                        size=(self.n_samples, 1)).astype('float32')
            uni_mjj_torch = torch.from_numpy(uni_mjj).reshape((-1, 1))
            my_cond_inputs = uni_mjj_torch.reshape((-1, 1))
        else: # sample mjj from actual data
            raw_train_mjj_vals = self.data_handler.inner_ANODE_datadict_train['labels']

            ## just copying actual mjj values (old approach)
            #train_size = raw_train_mjj_vals.shape[0]
            #train_mjj_vals = np.concatenate(
            #    [raw_train_mjj_vals for x in range(self.n_samples//train_size + 1)])
            #np.random.shuffle(train_mjj_vals)
            #train_mjj_vals = train_mjj_vals[:self.n_samples]

            ## fitting and sampling KDE
            mjj_logit = quick_logit(raw_train_mjj_vals.detach().cpu().numpy())
            train_mjj_vals = logit_transform_inverse(KernelDensity(
                bandwidth=self.KDE_bandwidth, kernel='gaussian').fit(
                    mjj_logit.reshape(-1, 1)).sample(self.n_samples),
                                                     max(raw_train_mjj_vals).item(),
                                                     min(raw_train_mjj_vals).item())

            train_mjj_vals = train_mjj_vals.astype(np.float32)         
            train_mjj_vals_torch = torch.from_numpy(train_mjj_vals).reshape((-1, 1))
            my_cond_inputs = train_mjj_vals_torch.reshape((-1, 1))

        # sample auxiliary variables with these conditionals
        outer_traindict = self.data_handler.outer_ANODE_datadict_train

        outer_samps_tensor_list = []
        n_samples_per_model = int(self.n_samples/len(self.model_list))
        for i, outer_model in enumerate(self.model_list):
            print(f"sampling from model {i+1}/{len(self.model_list)}")
            current_samps_tensor = outer_model.sample(
                num_samples=n_samples_per_model,
                cond_inputs=my_cond_inputs[i*n_samples_per_model:(i+1)*n_samples_per_model])

            if not self.no_mean_shift:
                # un-standardize it
                outer_samps_unstd = (current_samps_tensor*outer_traindict['std2']
                                     +outer_traindict['mean2']).detach().cpu().numpy()
            else:
                outer_samps_unstd = current_samps_tensor.detach().cpu().numpy()
            outer_samps_nans = np.argwhere(np.isinf(outer_samps_unstd))

            if not self.no_logit:
                # un-transform it
                current_samps_unstd_untrans = logit_transform_inverse(
                    outer_samps_unstd,
                    outer_traindict['max'].detach().cpu().numpy(),
                    outer_traindict['min'].detach().cpu().numpy()
                )
            else:
                current_samps_unstd_untrans = outer_samps_unstd *\
                    (outer_traindict['max'].detach().cpu().numpy() -\
                     outer_traindict['min'].detach().cpu().numpy())
                current_samps_unstd_untrans += outer_traindict['min'].detach().cpu().numpy()
            outer_samps_tensor_list.append(current_samps_unstd_untrans)

        outer_samps_unstd_untrans = np.concatenate(outer_samps_tensor_list)
        print("averaged the samples to a tensor of shape:", outer_samps_unstd_untrans.shape)

        # shuffle samples
        shuffled_indices = np.random.permutation(outer_samps_unstd_untrans.shape[0])
        shuffled_samples = outer_samps_unstd_untrans[shuffled_indices]
        shuffled_cond_inputs = my_cond_inputs.detach().cpu().numpy()[shuffled_indices]

        return shuffled_samples, shuffled_cond_inputs


    @torch.no_grad()
    def shift_samples(self, strength, constant_shift=False, random_shift=False,
                      shift_mj1=True, shift_dm=True, additional_shift=False):
        ## TODO add some explanation on the types of shifts
        ## TODO add some assert statements for stability
        assert self.sample_shift == 0, "Samples can only be shifted once!"
        assert not constant_shift, "Constant shift is not implemented, sorry!"
        assert not additional_shift, "Additional shift is not implemented, sorry!"
        self.sample_shift = strength

        # store actual copy as original samples
        self.original_sample_array = self.sample_array.copy()

        if random_shift:
            print("applying random shift...")
            raw_mjj_vals_inner = self.data_handler.innerdata_train[:, 0].copy()
            raw_mjj_vals_outer = self.data_handler.outerdata_train[:, 0].copy()
            raw_mjj_vals = np.concatenate(raw_mjj_vals_inner, raw_mjj_vals_outer)
            sample_shift = np.concatenate(
                [raw_mjj_vals for x in range(self.sample_array.shape[0]//raw_mjj_vals.shape[0] + 1)]
                )
            np.random.shuffle(sample_shift)
            sample_shift = sample_shift[:self.sample_array.shape[0]]
        else:
            print("applying artificial correlation with mjj...")
            sample_shift = self.cond_array.flatten()

        if shift_mj1:
            self.sample_array[:, 0] += strength*sample_shift 
        if shift_dm:
            self.sample_array[:, 1] += strength*sample_shift


    @torch.no_grad()
    def preprocess_samples(self, fiducial_cut=False, no_logit=False, no_mean_shift=False):
        ## does all the preprocessing for the ANODE data, including saving the min/max/mean/std

        self.fiducial_cut = fiducial_cut

        # the transformation will be based on the inner data
        reference_dict = self.data_handler.inner_ANODE_datadict_train

        if no_logit:
            if fiducial_cut:
                _, mask = logit_transform(
                    torch.from_numpy(self.sample_array).to(self.device),
                    reference_dict['max'],
                    reference_dict['min'], domain_cut=True, fiducial_cut=fiducial_cut
                )
                tensor2 = torch.from_numpy(self.sample_array).to(self.device)[mask]
            else:
                #tensor2 = torch.from_numpy(self.sample_array).to(self.device)
                #mask = (tensor2[:, 0] == tensor2[:, 0]) ## True
                tensor2 = torch.from_numpy(self.sample_array).to(self.device)
                tensor2 = (tensor2 - reference_dict['min'])/\
                    (reference_dict['max']-reference_dict['min'])
                mask = ((tensor2 >= 0.) & (tensor2 <= 1.)).all(axis=1)
                tensor2 = tensor2[mask]


        else:
            tensor2, mask = logit_transform(
                torch.from_numpy(self.sample_array).to(self.device), reference_dict['max'],
                reference_dict['min'], domain_cut=True, fiducial_cut=fiducial_cut
            )

        if not no_mean_shift:
            tensor2 = (tensor2 - reference_dict['mean2']) / reference_dict['std2']
        preprocessed_samples = tensor2.detach().cpu().numpy()
        np_mask = mask.detach().cpu().numpy()

        cond_array = self.cond_array.copy()[np_mask]
        nan_mask = np.argwhere(np.isnan(preprocessed_samples))

        preprocessed_samples = np.delete(preprocessed_samples, nan_mask, 0)
        cond_array = np.delete(cond_array, nan_mask, 0)

        self.preprocessed_samples = preprocessed_samples
        self.preprocessed_cond_array = cond_array
        self.masked_raw_samples = self.sample_array[np_mask]
        self.masked_raw_cond_array = self.cond_array[np_mask]


    def sanity_check(self, savefig=None, suppress_show=False):
        # plots dmj of samples and actual data in comparison
        # TODO could plot also mj1
        assert self.masked_raw_samples is not None, (
            "The samples first need to be preprocessed for proper masking!")

        inner_data = self.data_handler.inner_ANODE_datadict_train["tensor"].detach().cpu().numpy()
        inner_sigorbg = self.data_handler.inner_ANODE_datadict_train\
            ["sigorbg"].detach().cpu().numpy()
        plt.hist(inner_data[:, 1], range=[-0.1, 1], bins=22,
                 density=True, alpha=.5, label='INNER total training set')
        plt.hist(inner_data[inner_sigorbg == 0][:, 1], range=[-0.1, 1], bins=22,
                 alpha=.5, density=True, label='INNER bg, non-transformed training set')
        plt.hist(self.masked_raw_samples[:, 1], range=[-0.1, 1], bins=22, density=True,
                 alpha=.5, label='OUTER samps un-transformed and un-standardized')
        plt.legend(loc='upper right')
        plt.xlabel('mj2-mj1')
        plt.title('Untransformed and unstandardized samples vs training set')

        if savefig is not None:
            plt.savefig(savefig+".pdf", bbox_inches="tight")
        if not suppress_show:
            plt.show()
        plt.close()


    def sanity_check_after_cuts(self, savefig=None, suppress_show=False):
        # applying cuts to see visually how well the samples approximate the background
        assert self.masked_raw_samples is not None, (
            "The samples first need to be preprocessed for proper masking!")

        trained_outer_all = self.data_handler.outer_ANODE_datadict_train\
            ['tensor'].detach().cpu().numpy()
        trained_inner_all = self.data_handler.inner_ANODE_datadict_train\
            ['tensor'].detach().cpu().numpy()
        sample_array = self.masked_raw_samples

        # cuts
        samples_mask = (sample_array[:, 0] > 0.1) & (sample_array[:, 0] < 0.2) \
                     & (sample_array[:, 2] > 0.1) & (sample_array[:, 2] < 0.45)\
                     & (sample_array[:, 3] > 0.1) & (sample_array[:, 3] < 0.45)

        training_mask_in = (trained_inner_all[:, 0] > 0.1) & (trained_inner_all[:, 0] < 0.2) \
                         & (trained_inner_all[:, 2] > 0.1) & (trained_inner_all[:, 2] < 0.45)\
                         & (trained_inner_all[:, 3] > 0.1) & (trained_inner_all[:, 3] < 0.45)

        training_mask_out = (trained_outer_all[:, 0] > 0.1) & (trained_outer_all[:, 0] < 0.2) \
                          & (trained_outer_all[:, 2] > 0.1) & (trained_outer_all[:, 2] < 0.45)\
                          & (trained_outer_all[:, 3] > 0.1) & (trained_outer_all[:, 3] < 0.45)

        samples_cut = sample_array[samples_mask]
        training_cut_in = trained_inner_all[training_mask_in]
        training_cut_out = trained_outer_all[training_mask_out]

        sigorbg_in = self.data_handler.inner_ANODE_datadict_train['sigorbg'].detach().cpu().numpy()
        sigorbg_out = self.data_handler.outer_ANODE_datadict_train['sigorbg'].detach().cpu().numpy()

        training_in_cut_bg = trained_inner_all[training_mask_in & (sigorbg_in == 0)]
        training_out_cut_bg = trained_outer_all[training_mask_out & (sigorbg_out == 0)]

        plt.subplot(1, 2, 1)
        plt.rcParams["figure.figsize"] = (8, 6)
        plt.hist(training_cut_in[:, 1], range=[0, 1], bins=20, alpha=0.6, hatch='.',
                 fc=(1, 0, 0, 1), label='cut inner training set: all', zorder=2)
        plt.hist(training_in_cut_bg[:, 1], range=[0, 1], bins=20, alpha=0.4,
                 fc=(0.1, 0, 1, 1), label='cut inner training set: bg', zorder=3)
        plt.legend(loc='upper right')
        plt.xlabel('mj2-mj1')
        plt.title('Inner training set (cut): all vs bg')

        plt.subplot(1, 2, 2)
        plt.rcParams["figure.figsize"] = (8, 6)
        plt.hist(samples_cut[:, 1],
                 weights=np.ones(len(samples_cut))*len(training_in_cut_bg)/len(samples_cut),
                 range=[0, 1], bins=20, alpha=0.5, hatch='//', fc=(0.2, 0.8, 0, 0.5),
                 label='cut outer samps in SR', zorder=1)
        plt.hist(training_in_cut_bg[:, 1], range=[0, 1], bins=20, alpha=0.4,
                 fc=(0.1, 0, 1, 1), label='cut inner training set: bg', zorder=3)
        plt.legend(loc='upper right')
        plt.xlabel('mj2-mj1')
        plt.title('Outer samples in SR vs inner bg training set (cut)')

        plt.subplots_adjust(right=2.0)
        if savefig is not None:
            plt.savefig(savefig+".pdf", bbox_inches="tight")
        if not suppress_show:
            plt.show()
        plt.close()


## independent utility functions
def load_files(file_path):
    if isinstance(file_path, list):
        file_list = []
        for path in file_path:
            file_list.append(np.load(path).astype('float32'))
        loaded_file = np.vstack(file_list)
    else:
        loaded_file = np.load(file_path).astype('float32')
    return loaded_file


def apply_shift_to_dataset(dataset, strength, constant_shift=False, random_shift=False,
                           shift_mj1=True, shift_dm=True, mjj_vals=None, additional_shift=False):
    ## TODO add some assert statements for stability
    if constant_shift:
        shift = 3.5
        print("applying constant shift")
    elif random_shift:
        shift = mjj_vals.copy()
        np.random.shuffle(shift)
        shift = shift[:dataset.shape[0]]
        print("applying random shift")
    else:
        print("applying mjj-dependent shift")
        shift = dataset[:, 0]
    if additional_shift:
        print("stacking the shifted columns to the end")
        shifted_columns = dataset[:, [1, 2]]
        if shift_mj1:
            shifted_columns[:, 0] += strength*shift
        if shift_dm:
            shifted_columns[:, 1] += strength*shift
        ## appending additional columns to the back
        dataset = np.hstack((dataset[:, :-1], shifted_columns, dataset[:, -1:]))
    else:
        if shift_mj1:
            dataset[:, 1] += strength*shift
        if shift_dm:
            dataset[:, 2] += strength*shift
    return dataset


def logit_transform(data, datamax, datamin, domain_cut=False, fiducial_cut=False):

    data2 = (data-datamin)/(datamax-datamin)

    if fiducial_cut:
        mask = (data2[:, 0] > 0.05) & (data2[:, 0] < 0.95) &\
            (data2[:, 1] > 0.05) & (data2[:, 1] < 0.95) &\
            (data2[:, 2] > 0.05) & (data2[:, 2] < 0.95) &\
            (data2[:, 3] > 0.05) & (data2[:, 3] < 0.95)
    elif domain_cut:
        mask = (data2[:, 0] > 0) & (data2[:, 0] < 1) &\
            (data2[:, 1] > 0) & (data2[:, 1] < 1) &\
            (data2[:, 2] > 0) & (data2[:, 2] < 1) &\
            (data2[:, 3] > 0) & (data2[:, 3] < 1)
    else:
        mask = (data2[:, 0] == data2[:, 0]) ## True

    data3 = data2[mask]
    data4 = torch.log((data3)/(1-data3))
    return data4, mask


def logit_transform_inverse(data, datamax, datamin):

    dataout = (datamin + datamax*np.exp(data))/(1 + np.exp(data))

    return dataout


def quick_logit(x):
    x_norm = (x-min(x))/(max(x)-min(x))
    x_norm = x_norm[(x_norm != 0) & (x_norm != 1)]
    logit = np.log(x_norm/(1-x_norm))
    logit = logit[~np.isnan(logit)]
    return logit


def create_dataset(data, device=torch.device("cpu")):
    sigorbg = torch.from_numpy(data[:, -1]).to(device)
    labels = torch.from_numpy(data[:, 0:1]).to(device)
    tensor = torch.from_numpy(data[:, 1:-1]).to(device)
    return sigorbg, labels, tensor


def load_dataset(data, batch_size=256, external_datadict=None, fiducial_cut=False,
                 shuffle_loader=False, cond_range=None, no_logit=False, device=torch.device("cpu"),
                 no_mean_shift=False):
    ## General dataset loading function. The training data should be loaded without external
    ##   datadict fiducial cut. The test data needs the trainind datadict given and the
    ##   fiducial cut is optional. If cond_range is provided with a tuple, events with mjj
    ##   outside this range will be cut off first, before computing masks or min/max/std/mean.
    ##   no_mean_shift disables shifting mean=0 and std=1

    if cond_range is not None:
        assert len(cond_range) == 2, "cond_range must be 2-element array-like object!"
        cond_mask = np.logical_and((data[:, 0] > cond_range[0]), (data[:, 0] < cond_range[1]))
        input_data = data[cond_mask]
    else:
        input_data = data

    datadict = {}

    datadict['sigmask'] = torch.from_numpy(input_data[:, -1] == 1)
    datadict['bgmask'] = torch.from_numpy(input_data[:, -1] == 0)

    datadict['sigorbg'], datadict['labels'], datadict['tensor'] = create_dataset(input_data,
                                                                                 device=device)

    if external_datadict is not None:
        datadict['max'] = external_datadict['max'].clone()
        datadict['min'] = external_datadict['min'].clone()
    else:
        datadict['max'] = torch.max(datadict['tensor'], dim=0).values
        datadict['min'] = torch.min(datadict['tensor'], dim=0).values

    if no_logit:
        if fiducial_cut:
            _, mask = logit_transform(
                datadict['tensor'], datadict['max'], datadict['min'],
                domain_cut=True, fiducial_cut=fiducial_cut
            )
            tensor2 = datadict['tensor'][mask]
        else:
            #tensor2 = datadict['tensor']
            #mask = (tensor2[:,0] == tensor2[:,0]) ## True

            # data should always be in [0, 1]
            tensor2 = (datadict['tensor']-datadict['min'])/(datadict['max']-datadict['min'])
            mask = ((tensor2 > 0.) & (tensor2 < 1.)).all(axis=1)
            tensor2 = tensor2[mask]
    else:
        tensor2, mask = logit_transform(
            datadict['tensor'], datadict['max'], datadict['min'],
            domain_cut=True, fiducial_cut=fiducial_cut
        )


    if external_datadict is not None:
        datadict['mean2'] = external_datadict['mean2'].clone()
        datadict['std2'] = external_datadict['std2'].clone()
        # to correct the loss when no_logit is used:
        datadict['std2_logit_fix'] = external_datadict['std2_logit_fix'].clone()
    else:
        # making sure the fiducial cut doesn't affect the std and mean
        if fiducial_cut:
            # reference_tensor: tensor of desired dataset
            # reference_tensor_logit_fix: tensor without logit
            #           (always needed for no-logit loss correction)
            reference_tensor_logit_fix, _ = logit_transform(
                datadict['tensor'], datadict['max'], datadict['min'],
                domain_cut=True, fiducial_cut=False
            )
            if no_logit:
                reference_tensor = datadict['tensor']
            else:
                reference_tensor = reference_tensor_logit_fix
        else:
            reference_tensor_logit_fix, _ = logit_transform(
                datadict['tensor'], datadict['max'], datadict['min'],
                domain_cut=True, fiducial_cut=fiducial_cut
            )
            reference_tensor = tensor2

        datadict['mean2'] = torch.mean(reference_tensor, dim=0)
        datadict['std2'] = torch.std(reference_tensor, dim=0)
        datadict['std2_logit_fix'] = torch.std(reference_tensor_logit_fix, dim=0)

    datadict['sigmask'] = datadict['sigmask'][mask]
    datadict['bgmask'] = datadict['bgmask'][mask]
    datadict['sigorbg'] = datadict['sigorbg'][mask]
    datadict['labels'] = datadict['labels'][mask]
    datadict['tensor'] = datadict['tensor'][mask]
    datadict['mask'] = mask

    if not no_mean_shift:
        datadict['tensor2'] = (tensor2 - datadict['mean2']) / datadict['std2']
    else:
        datadict['tensor2'] = tensor2
    datadict['dataset'] = torch.utils.data.TensorDataset(datadict['tensor2'], datadict['labels'])
    datadict['loader'] = torch.utils.data.DataLoader(
        datadict['dataset'], batch_size=batch_size, shuffle=shuffle_loader)

    return datadict


def stack_data(data_array, cond_labels, sig_labels=None, samples=False, CWoLa_outer=False):
    # Stack data with its labels to make a 7-column array.
    # Form of data is mjj, mj1...tau2, dataorsample, sigorbg

    if CWoLa_outer:
        assert sig_labels is not None, "Valid signal or background labels need to be provided!"
        data_or_sample = np.zeros((data_array.shape[0], 1))
    elif samples:
        data_or_sample = np.zeros((data_array.shape[0], 1))
        sig_labels = 5*np.ones((data_array.shape[0], 1))
    else:
        assert sig_labels is not None, "Valid signal or background labels need to be provided!"
        data_or_sample = np.ones((data_array.shape[0], 1))

    stacked_data = np.hstack((cond_labels.reshape((-1, 1)), data_array, data_or_sample,
                              sig_labels.reshape((-1, 1)))).astype('float32')

    # cleaning out infs and nans, to be sure nothing weird has happened meanwhile
    stacked_data = np.delete(stacked_data, np.argwhere(np.isinf(stacked_data)), 0)
    stacked_data = np.delete(stacked_data, np.argwhere(np.isnan(stacked_data)), 0)

    return stacked_data


def mix_data_samples(data_handler, samples_handler=None, oversampling=False, CWoLa=False,
                     supervised=False, idealized_AD=False, savedir=None, separate_val_set=False):
    # Take preprocessed SR data and samples (or SB data) and mix them for classifier training.
    #   If CWoLa is set true, SR and SB data will be mixed from the data_handler.
    #   Give a directory path to savedir if the arrays should be saved right away.

    if CWoLa or supervised or idealized_AD:
        if samples_handler is not None:
            print("No samples handler should be given if CWoLa or supervised is set true!")
            return None
        if CWoLa:
            assert data_handler.inner_CWoLa_datadict_train is not None, (
                "Need to call the CWoLa data preproceesing first!")
    else:
        if samples_handler is None:
            print("A samples handler needs to be given unless CWoLa is set true!")
            return None

    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)

    if CWoLa:
        # stacking each individually with additional information
        SR_train_data_array = data_handler.inner_CWoLa_datadict_train\
            ['tensor2'].detach().cpu().numpy()
        SR_train_cond_array = data_handler.inner_CWoLa_datadict_train\
            ['labels'].detach().cpu().numpy()
        SR_train_sig_labels = data_handler.inner_CWoLa_datadict_train\
            ['sigorbg'].detach().cpu().numpy()

        SR_test_data_array = data_handler.inner_CWoLa_datadict_test\
            ['tensor2'].detach().cpu().numpy()
        SR_test_cond_array = data_handler.inner_CWoLa_datadict_test\
            ['labels'].detach().cpu().numpy()
        SR_test_sig_labels = data_handler.inner_CWoLa_datadict_test\
            ['sigorbg'].detach().cpu().numpy()

        if data_handler.innerdata_extrasig is not None:
            SR_extr_data_array = data_handler.inner_CWoLa_datadict_extrasig\
                ['tensor2'].detach().cpu().numpy()
            SR_extr_cond_array = data_handler.inner_CWoLa_datadict_extrasig\
                ['labels'].detach().cpu().numpy()
            SR_extr_sig_labels = np.ones(SR_extr_cond_array.shape) 

        if data_handler.innerdata_val is not None:
            SR_val_data_array = data_handler.inner_CWoLa_datadict_val\
                ['tensor2'].detach().cpu().numpy()
            SR_val_cond_array = data_handler.inner_CWoLa_datadict_val\
                ['labels'].detach().cpu().numpy()
            SR_val_sig_labels = data_handler.inner_CWoLa_datadict_val\
                ['sigorbg'].detach().cpu().numpy()

        SB_train_data_array = data_handler.outer_CWoLa_datadict_train\
            ['tensor2'].detach().cpu().numpy()
        SB_train_cond_array = data_handler.outer_CWoLa_datadict_train\
            ['labels'].detach().cpu().numpy()
        SB_train_sig_labels = data_handler.outer_CWoLa_datadict_train\
            ['sigorbg'].detach().cpu().numpy()

        SB_test_data_array = data_handler.outer_CWoLa_datadict_test\
            ['tensor2'].detach().cpu().numpy()
        SB_test_cond_array = data_handler.outer_CWoLa_datadict_test\
            ['labels'].detach().cpu().numpy()
        SB_test_sig_labels = data_handler.outer_CWoLa_datadict_test\
            ['sigorbg'].detach().cpu().numpy()

        stacked_SR_data_train = stack_data(SR_train_data_array, SR_train_cond_array,
                                           sig_labels=SR_train_sig_labels, CWoLa_outer=False)
        stacked_SR_data_test = stack_data(SR_test_data_array, SR_test_cond_array,
                                          sig_labels=SR_test_sig_labels, CWoLa_outer=False)
        if data_handler.innerdata_extrasig is not None:
            stacked_SR_data_extr = stack_data(SR_extr_data_array, SR_extr_cond_array,
                                              sig_labels=SR_extr_sig_labels, CWoLa_outer=False)
        if data_handler.innerdata_val is not None:
            stacked_SR_data_val = stack_data(SR_val_data_array, SR_val_cond_array,
                                             sig_labels=SR_val_sig_labels, CWoLa_outer=False)

        stacked_SB_data_train = stack_data(SB_train_data_array, SB_train_cond_array,
                                           sig_labels=SB_train_sig_labels, CWoLa_outer=True)
        stacked_SB_data_test = stack_data(SB_test_data_array, SB_test_cond_array,
                                          sig_labels=SB_test_sig_labels, CWoLa_outer=True)

        # distribute outer train and "test" data proportionally
        stacked_SB_data_full = np.concatenate((stacked_SB_data_train, stacked_SB_data_test))

        if oversampling:
            if data_handler.innerdata_val is not None:
                n_data_full = stacked_SR_data_train.shape[0] + stacked_SR_data_val.shape[0]
                train_sample_fraction = stacked_SR_data_train.shape[0] / n_data_full
                n_samps_train = int(train_sample_fraction * stacked_SB_data_full.shape[0])
                val_events = np.concatenate((stacked_SB_data_full[n_samps_train:],
                                             stacked_SR_data_val)).astype('float32')
                test_events = stacked_SR_data_test.astype('float32')
            else:
                n_data_full = stacked_SR_data_train.shape[0] + stacked_SR_data_train.shape[0]
                train_sample_fraction = stacked_SR_data_train.shape[0] / n_data_full
                n_samps_train = int(train_sample_fraction * stacked_SB_data_full.shape[0])
                test_events = np.concatenate((stacked_SB_data_full[n_samps_train:],
                                              stacked_SR_data_test)).astype('float32')
            train_events = np.concatenate((stacked_SB_data_full[:n_samps_train],
                                           stacked_SR_data_train)).astype('float32')
            #train_events = np.concatenate(
            #    (stacked_SB_data_train, stacked_SR_data_train)).astype('float32')
            #test_events = np.concatenate(
            #    (stacked_SB_data_test, stacked_SR_data_test)).astype('float32')

        else:
            if data_handler.innerdata_val is not None:
                n_data_full = stacked_SR_data_train.shape[0] + stacked_SR_data_val.shape[0]
                train_sample_fraction = stacked_SR_data_train.shape[0] / n_data_full
                n_samps_train = int(train_sample_fraction * stacked_SB_data_full.shape[0])
                n_samps_val = stacked_SB_data_full.shape[0] - n_samps_train

                n_train = min(n_samps_train, stacked_SR_data_train.shape[0])
                n_val = min(n_samps_val, stacked_SR_data_val.shape[0])
                val_events = np.concatenate((stacked_SB_data_full[-n_val:],
                                              stacked_SR_data_val[:n_val])).astype('float32')
                test_events = stacked_SR_data_test.astype('float32')

            else:
                n_data_full = stacked_SR_data_train.shape[0] + stacked_SR_data_train.shape[0]
                train_sample_fraction = stacked_SR_data_train.shape[0] / n_data_full
                n_samps_train = int(train_sample_fraction * stacked_SB_data_full.shape[0])
                n_samps_test = stacked_SB_data_full.shape[0] - n_samps_train

                n_train = min(n_samps_train, stacked_SR_data_train.shape[0])
                n_test = min(n_samps_test, stacked_SR_data_test.shape[0])
                test_events = np.concatenate((stacked_SB_data_full[-n_test:],
                                              stacked_SR_data_test[:n_test])).astype('float32')

            train_events = np.concatenate((stacked_SB_data_full[:n_train],
                                           stacked_SR_data_train[:n_train])).astype('float32')

            #n_train = min(stacked_SB_data_train.shape[0], stacked_SR_data_train.shape[0])
            #n_test = min(stacked_SB_data_test.shape[0], stacked_SR_data_test.shape[0])
            #train_events = np.concatenate((stacked_SB_data_train[:n_train],
            #                               stacked_SR_data_train[:n_train])).astype('float32')
            #test_events = np.concatenate((stacked_SB_data_test[:n_test],
            #                              stacked_SR_data_test[:n_test])).astype('float32')

    else:
        # stacking each individually with additional information
        SR_train_data_array = data_handler.inner_ANODE_datadict_train\
            ['tensor2'].detach().cpu().numpy()
        SR_train_cond_array = data_handler.inner_ANODE_datadict_train\
            ['labels'].detach().cpu().numpy()
        SR_train_sig_labels = data_handler.inner_ANODE_datadict_train\
            ['sigorbg'].detach().cpu().numpy()

        SR_test_data_array = data_handler.inner_ANODE_datadict_test\
            ['tensor2'].detach().cpu().numpy()
        SR_test_cond_array = data_handler.inner_ANODE_datadict_test\
            ['labels'].detach().cpu().numpy()
        SR_test_sig_labels = data_handler.inner_ANODE_datadict_test\
            ['sigorbg'].detach().cpu().numpy()

        if data_handler.innerdata_extrasig is not None:
            SR_extr_data_array = data_handler.inner_ANODE_datadict_extrasig\
                ['tensor2'].detach().cpu().numpy()
            SR_extr_cond_array = data_handler.inner_ANODE_datadict_extrasig\
                ['labels'].detach().cpu().numpy()
            SR_extr_sig_labels = np.ones(SR_extr_cond_array.shape)

        if data_handler.innerdata_val is not None:
            SR_val_data_array = data_handler.inner_ANODE_datadict_val\
                ['tensor2'].detach().cpu().numpy()
            SR_val_cond_array = data_handler.inner_ANODE_datadict_val\
                ['labels'].detach().cpu().numpy()
            SR_val_sig_labels = data_handler.inner_ANODE_datadict_val\
                ['sigorbg'].detach().cpu().numpy()

        if data_handler.inner_ANODE_datadict_extrabkg is not None:
            print("Using extra background...")
            SR_extr_bkg_data_array = data_handler.inner_ANODE_datadict_extrabkg\
                ['tensor2'].detach().cpu().numpy()
            SR_extr_bkg_cond_array = data_handler.inner_ANODE_datadict_extrabkg\
                ['labels'].detach().cpu().numpy()
            SR_extr_bkg_sig_labels = np.zeros(SR_extr_bkg_cond_array.shape)
            stacked_SR_data_extr_bkg = stack_data(SR_extr_bkg_data_array, SR_extr_bkg_cond_array,
                                              sig_labels=SR_extr_bkg_sig_labels, samples=False)

        if not supervised and not idealized_AD:
            sample_data_array = samples_handler.preprocessed_samples
            sample_cond_array = samples_handler.preprocessed_cond_array

        stacked_SR_data_train = stack_data(SR_train_data_array, SR_train_cond_array,
                                           sig_labels=SR_train_sig_labels, samples=False)
        stacked_SR_data_test = stack_data(SR_test_data_array, SR_test_cond_array,
                                          sig_labels=SR_test_sig_labels, samples=False)
        if data_handler.innerdata_extrasig is not None:
            stacked_SR_data_extr = stack_data(SR_extr_data_array, SR_extr_cond_array,
                                              sig_labels=SR_extr_sig_labels, samples=False)

        if data_handler.innerdata_val is not None:
            stacked_SR_data_val = stack_data(SR_val_data_array, SR_val_cond_array,
                                             sig_labels=SR_val_sig_labels, samples=False)
           
        if not supervised and not idealized_AD:
            stacked_samples = stack_data(sample_data_array, sample_cond_array,
                                         sig_labels=None, samples=True)

            # stacking data and samples together
            if oversampling:
                # using the full sample number, distributing them proportionally do test/train sets
                print("using the full number of samples...")
                if data_handler.innerdata_val is not None:
                    n_data_full = stacked_SR_data_train.shape[0] + stacked_SR_data_val.shape[0]
                    train_sample_fraction = stacked_SR_data_train.shape[0] / n_data_full
                    n_samps_train = int(train_sample_fraction * stacked_samples.shape[0])
                    val_events = np.concatenate((stacked_samples[n_samps_train:],
                                                 stacked_SR_data_val)).astype('float32')
                    test_events = stacked_SR_data_test.astype('float32')

                    #n_data_full = stacked_SR_data_train.shape[0] + stacked_SR_data_test.shape[0] + stacked_SR_data_val.shape[0]
                    #train_sample_fraction = stacked_SR_data_train.shape[0] / n_data_full
                    #test_sample_fraction = stacked_SR_data_test.shape[0] / n_data_full
                    #val_sample_fraction = stacked_SR_data_val.shape[0] / n_data_full
                    #n_samps_train = int(train_sample_fraction * stacked_samples.shape[0])
                    #n_samps_test = int(test_sample_fraction * stacked_samples.shape[0])
                    #n_samps_val = stacked_samples.shape[0]-n_samps_train-n_samps_test
                    #val_events = np.concatenate((
                    #    stacked_samples[n_samps_train+n_samps_test:n_samps_train+n_samps_test+n_samps_val],
                    #    stacked_SR_data_val)).astype('float32')
                   
                else:
                    n_data_full = stacked_SR_data_train.shape[0] + stacked_SR_data_test.shape[0]
                    train_sample_fraction = stacked_SR_data_train.shape[0] / n_data_full
                    n_samps_train = int(train_sample_fraction * stacked_samples.shape[0])
                    test_events = np.concatenate((stacked_samples[n_samps_train:],
                                                  stacked_SR_data_test)).astype('float32')

                train_events = np.concatenate((stacked_samples[:n_samps_train],
                                               stacked_SR_data_train)).astype('float32')
            else:
                # mixing test and train samples each with the same number of samples as there are data
                print("skimming down sample number to match corresponding data size...")
                train_events = np.concatenate(
                    (stacked_samples[:stacked_SR_data_train.shape[0]],
                     stacked_SR_data_train)).astype('float32')
                if data_handler.innerdata_val is not None:
                    val_events = np.concatenate(
                       (stacked_samples[stacked_SR_data_val.shape[0]:],
                        stacked_SR_data_val)).astype('float32')
                    test_events = stacked_SR_data_test.astype('float32')
                else:
                    test_events = np.concatenate(
                        (stacked_samples[-stacked_SR_data_test.shape[0]:],
                         stacked_SR_data_test)).astype('float32')

        else:
            if supervised:
                train_events = stacked_SR_data_train.astype('float32')
                test_events = stacked_SR_data_test.astype('float32')
                if data_handler.innerdata_val is not None:
                    val_events = stacked_SR_data_val.astype('float32')
            elif idealized_AD:
                if data_handler.inner_ANODE_datadict_extrabkg is not None:
                    if data_handler.innerdata_val is not None:
                        n_data_full = stacked_SR_data_train.shape[0] + stacked_SR_data_val.shape[0]
                        train_fraction = stacked_SR_data_train.shape[0] / n_data_full
                        n_extra_bkg_train = int(train_fraction * stacked_SR_data_extr_bkg.shape[0])

                        val_ideal_samples = stacked_SR_data_extr_bkg[n_extra_bkg_train:]
                        val_ideal_samples[:,-2] = np.zeros(val_ideal_samples.shape[0])
                        val_events = np.concatenate((stacked_SR_data_val,
                                                     val_ideal_samples)).astype('float32')
                        test_events = stacked_SR_data_test.astype('float32')

                    else:
                        n_data_full = stacked_SR_data_train.shape[0] + stacked_SR_data_test.shape[0]
                        train_fraction = stacked_SR_data_train.shape[0] / n_data_full
                        n_extra_bkg_train = int(train_fraction * stacked_SR_data_extr_bkg.shape[0])

                        test_ideal_samples = stacked_SR_data_extr_bkg[n_extra_bkg_train:]
                        test_ideal_samples[:,-2] = np.zeros(test_ideal_samples.shape[0])
                        test_events = np.concatenate((stacked_SR_data_test,
                                                       test_ideal_samples)).astype('float32')

                    train_ideal_samples = stacked_SR_data_extr_bkg[:n_extra_bkg_train]
                    train_ideal_samples[:,-2] = np.zeros(train_ideal_samples.shape[0])
                    train_events = np.concatenate((stacked_SR_data_train,
                                                   train_ideal_samples)).astype('float32')

                else:
                    if data_handler.innerdata_val is not None:
                        n_val_samples = int(stacked_SR_data_test.shape[0]/2)
                        val_ideal_samples = stacked_SR_data_test[:n_val_samples]
                        val_ideal_samples = val_ideal_samples[val_ideal_samples[:,-1]==0]
                        val_ideal_samples[:,-2] = np.zeros(val_ideal_samples.shape[0])
                        val_events = np.concatenate((stacked_SR_data_val[n_val_samples:],
                                                      val_ideal_samples)).astype('float32') 
                        test_events = stacked_SR_data_test.astype('float32') 
                    else:
                        n_test_samples = int(stacked_SR_data_test.shape[0]/2)
                        test_ideal_samples = stacked_SR_data_test[:n_test_samples]
                        test_ideal_samples = test_ideal_samples[test_ideal_samples[:,-1]==0]
                        test_ideal_samples[:,-2] = np.zeros(test_ideal_samples.shape[0])
                        test_events = np.concatenate((stacked_SR_data_test[n_test_samples:],
                                                      test_ideal_samples)).astype('float32') 

                    n_train_samples = int(stacked_SR_data_train.shape[0]/2)
                    train_ideal_samples = stacked_SR_data_train[:n_train_samples]
                    train_ideal_samples = train_ideal_samples[train_ideal_samples[:,-1]==0]
                    train_ideal_samples[:,-2] = np.zeros(train_ideal_samples.shape[0])
                    train_events = np.concatenate((stacked_SR_data_train[n_train_samples:],
                                                   train_ideal_samples)).astype('float32') 


    if data_handler.innerdata_extrasig is not None:
        extr_events = stacked_SR_data_extr.astype('float32')
        #if supervised:
        #    n_data_full = train_events.shape[0] + test_events.shape[0]
        #    train_fraction = train_events.shape[0] / n_data_full
        #    n_signal_train = int(train_fraction * extr_events.shape[0])
        #    train_events = np.vstack((train_events, extr_events[:n_signal_train]))
        #    test_events = np.vstack((test_events, extr_events[n_signal_train:]))
    
    if supervised and data_handler.inner_ANODE_datadict_extrabkg is not None:
        n_data_full = train_events.shape[0] + test_events.shape[0]
        train_fraction = train_events.shape[0] / n_data_full
        n_extra_bkg_train = int(train_fraction * stacked_SR_data_extr_bkg.shape[0])
        train_events = np.vstack((train_events, stacked_SR_data_extr_bkg[:n_extra_bkg_train]))
        if data_handler.innerdata_val is None:
            test_events = np.vstack((test_events, stacked_SR_data_extr_bkg[n_extra_bkg_train:]))

    # shuffle rows
    np.random.seed(42)
    np.random.shuffle(train_events)
    np.random.shuffle(test_events)

    X_train = train_events
    X_test = test_events

    y_train = train_events[:, -2]
    y_test = test_events[:, -2]

    if (supervised or separate_val_set) and data_handler.innerdata_val is not None:
        np.random.shuffle(val_events)
        X_val = val_events
        y_val = val_events[:, -2]

    if data_handler.innerdata_extrasig is not None:
        np.random.shuffle(extr_events)
        X_extrasig = extr_events
        y_extrasig = extr_events[:, -2]
    else:
        X_extrasig = None
        y_extrasig = None

    # split validation set from test data
    if (supervised or separate_val_set) and data_handler.innerdata_val is None:
        n_val = int(2./5*X_test.shape[0])
        X_val = X_test[:n_val]
        y_val = X_test[:n_val]
        X_test = X_test[n_val:]
        y_test = y_test[n_val:]

    # save train and test sets
    if savedir is not None:
        np.save(os.path.join(savedir, 'X_train.npy'), X_train)
        np.save(os.path.join(savedir, 'y_train.npy'), y_train)
        np.save(os.path.join(savedir, 'X_test.npy'), X_test)
        np.save(os.path.join(savedir, 'y_test.npy'), y_test)
        np.save(os.path.join(savedir, 'X_extrasig.npy'), X_extrasig)
        np.save(os.path.join(savedir, 'y_extrasig.npy'), y_extrasig)
        if supervised or separate_val_set:
            np.save(os.path.join(savedir, 'X_validation.npy'), X_val)
            np.save(os.path.join(savedir, 'y_validation.npy'), y_val)

    if not supervised and not separate_val_set:
        return X_train, y_train, X_test, y_test, X_extrasig, y_extrasig
    else:
        return X_train, y_train, X_test, y_test, X_val, y_val


def plot_data_sample_comparison(X_vals, y_vals, nbins=50, alpha=0.5, title=None, savefig=None,
                                suppress_show=False, data_label=None, sample_label=None, 
                                data_color=None, sample_color=None, step_hist=False,
                                unstandardize=None, draw_signal=False, signal_color=None):
    # sanity check plot comparing data and samples before running the classifier

    if data_label is None:
        data_label = "Data"
    if sample_label is None:
        sample_label = "Samples"
    if data_color is None:
        data_color = "tab:blue"
    if sample_color is None:
        sample_color = "tab:orange"
    if signal_color is None:
        signal_color = "magenta"

    sample_hist_type = "step" if step_hist else "bar"

    data_array = X_vals[y_vals == 1]
    samp_array = X_vals[y_vals == 0]

    if unstandardize is not None:
        reference_data = unstandardize[:,1:-1]
        mean_vals = np.mean(reference_data, axis=0)
        std_vals = np.std(reference_data, axis=0)
        data_array[:,1:5] = data_array[:,1:5]*std_vals+mean_vals
        samp_array[:,1:5] = samp_array[:,1:5]*std_vals+mean_vals
        tau_unit = ""
        m_unit = " (TeV)"
    else:
        tau_unit = " (std.)"
        m_unit = " (std.)"
       
    signal_array = data_array[data_array[:,-1]==1]
    data_array = data_array[data_array[:,-1]==0]

    fig = plt.figure(figsize=(6,8))
    gs = gridspec.GridSpec(6,4)
    #plt.subplot(3, 2, 1)
    plt.subplot(gs[:2,:2])
    _, binning, _ = plt.hist(data_array[:, 1], bins=nbins, density=True, label=data_label,
                             color=data_color, alpha=alpha)
    if len(samp_array) != 0:
        plt.hist(samp_array[:, 1], bins=binning, density=True, label=sample_label,
                 color=sample_color, alpha=alpha, histtype=sample_hist_type)
    if draw_signal:
        plt.hist(signal_array[:, 1], bins=binning, density=True, label="Signal",
                             color=signal_color, alpha=alpha, histtype=sample_hist_type)
    plt.xlabel(r'$m_{J1}$'+m_unit)
    plt.ylabel("Events (a.u.)")
    plt.legend(loc='upper right', frameon=False)

    #plt.subplot(3, 2, 2)
    plt.subplot(gs[:2,2:])
    _, binning, _ = plt.hist(data_array[:, 2], bins=nbins, density=True, label=data_label,
                             color=data_color, alpha=alpha)
    if len(samp_array) != 0:
        plt.hist(samp_array[:, 2], bins=binning, density=True, label=sample_label,
                 color=sample_color, alpha=alpha, histtype=sample_hist_type)
    if draw_signal:
        plt.hist(signal_array[:, 2], bins=binning, density=True, label="Signal",
                             color=signal_color, alpha=alpha, histtype=sample_hist_type)
    plt.xlabel(r'$m_{J2}-m_{J1}$'+m_unit)
    plt.ylabel("Events (a.u.)")
    plt.legend(loc='upper right', frameon=False)

    #plt.subplot(3, 2, 3)
    plt.subplot(gs[2:4,:2])
    _, binning, _ = plt.hist(data_array[:, 3], bins=nbins, density=True, label=data_label,
                             color=data_color, alpha=alpha)
    if len(samp_array) != 0:
        plt.hist(samp_array[:, 3], bins=binning, density=True, label=sample_label,
                 color=sample_color, alpha=alpha, histtype=sample_hist_type)
    if draw_signal:
        plt.hist(signal_array[:, 3], bins=binning, density=True, label="Signal",
                             color=signal_color, alpha=alpha, histtype=sample_hist_type)
    plt.xlabel(r'$\tau_{21}^{J1}$'+tau_unit)
    plt.ylabel("Events (a.u.)")
    plt.legend(loc='upper left', frameon=False)
    plt.ylim(top=1.2*plt.gca().get_ylim()[1])

    #plt.subplot(3, 2, 4)
    plt.subplot(gs[2:4,2:])
    _, binning, _ = plt.hist(data_array[:, 4], bins=nbins, density=True, label=data_label,
                             color=data_color, alpha=alpha)
    if len(samp_array) != 0:
        plt.hist(samp_array[:, 4], bins=binning, density=True, label=sample_label,
                 color=sample_color, alpha=alpha, histtype=sample_hist_type)
    if draw_signal:
        plt.hist(signal_array[:, 4], bins=binning, density=True, label="Signal",
                             color=signal_color, alpha=alpha, histtype=sample_hist_type)
    plt.xlabel(r'$\tau_{21}^{J2}$'+tau_unit)
    plt.ylabel("Events (a.u.)")
    plt.legend(loc='upper left', frameon=False)
    plt.ylim(top=1.25*plt.gca().get_ylim()[1])

    #plt.subplot(3, 2, 5)
    plt.subplot(gs[4:,1:3])
    _, binning, _ = plt.hist(data_array[:, 0], bins=nbins, density=True, label=data_label,
                             color=data_color, alpha=alpha)
    if len(samp_array) != 0:
        plt.hist(samp_array[:, 0], bins=binning, density=True, label=sample_label,
                 color=sample_color, alpha=alpha, histtype=sample_hist_type)
    if draw_signal:
        plt.hist(signal_array[:, 0], bins=binning, density=True, label="Signal",
                             color=signal_color, alpha=alpha, histtype=sample_hist_type)
    plt.xlabel(r'$m_{JJ}$ (TeV)')
    plt.ylabel("Events (a.u.)")
    plt.legend(loc='upper right', frameon=False)
    plt.ylim(top=1.05*plt.gca().get_ylim()[1])

    if title is not None:
        plt.suptitle(title)

    fig.tight_layout()

    if savefig is not None:
        plt.savefig(savefig+".pdf", bbox_inches="tight")
    if not suppress_show:
        plt.show()
    plt.close()

    ## maybe implement later TODO
    #plt.subplot(1,3,1)
    #plt.hist(train_data[:,2],range=[0.2,0.6],bins=20,density=True,label='train data',alpha=0.5)
    #plt.hist(samps[:,2],range=[0.2,0.6],bins=20,density=True,label='samps',alpha=0.5)
    #plt.xlabel('mj2-mj1')
    #plt.legend(loc='upper right')
    #plt.title('Comparing train data and samps, normalized')
    #
    #plt.subplot(1,3,2)
    #plt.hist(train_data[:,2],range=[0.2,0.6],bins=20,label='all train data in SR',alpha=0.5)
    #plt.hist(train_data[:,2][train_data[:,-1] == 0],range=[0.2,0.6],bins=20,alpha=0.5,label='bg train data in SR')
    #plt.hist(train_data[:,2][train_data[:,-1] == 1],range=[0.2,0.6],bins=20,alpha=0.8,label='signal train data in SR')
    ##plt.yscale('log')
    #plt.xlabel('mj2-mj1')
    #plt.legend(loc='upper right')
    #plt.title('train data bg and signal')
    #
    #plt.subplot(1,3,3)
    #plt.hist(train_data[:,2][train_data[:,-1] == 0],range=[0.2,0.6],bins=20,density=True,alpha=0.5,label='bg train data in SR')
    #plt.hist(samps[:,2],range=[0.2,0.6],bins=20,density=True,label='samps',alpha=0.5)
    ##plt.yscale('log')
    #plt.xlabel('mj2-mj1')
    #plt.legend(loc='upper right')
    #plt.title('Comparing train data bg and samps, normalized')
    #
    #plt.subplots_adjust(right=2.0)
    #
    #plt.show()
