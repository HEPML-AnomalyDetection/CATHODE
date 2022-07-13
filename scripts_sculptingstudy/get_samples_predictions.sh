#!/bin/bash

# use density estimator to generate samples in the signal region (as many as there are "data" events) and then get the predictions of both data events and samples.
# Afterwards, select data events according to pre-defined set of selection efficiencies (20%, 5%, 1% 0.1%) and cut artificial samples at the same **thresholds**.
# The mjj values of such selected events of both data and artificial samples are stored in separate numpy files, called "mjj_data_eff[efficiency].npy" for data
# and "mjj_samps_eff[efficiency].npy" for the samples, respectively. The [efficiency] is a placeholder and stands for the respective selection efficiencies' floating point value
# (i.e. 0.2 for 20% and so on).

for mass in {3200..4400..200}
do
    python get_samples_predictions.py --data-dir ../separated_data_sculptingstudy/sr_$mass --out-dir ../results_sculptingstudy/sr_m$mass --result-DE-dir ../results_sculptingstudy/sr_m$mass --result-clsf-dir ../results_sculptingstudy/sr_m$mass --DE-config ../DE_MAF_model.yml
done
