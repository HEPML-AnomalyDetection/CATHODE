#!/bin/bash

## create directory that will contain results
mkdir ../results_sculptingstudy

## run trainings for all signal regions
for mass in {3200..4400..200}
do
    python3 ../run_all.py --data_dir ../separated_data_sculptingstudy/sr_${mass}/ --save_dir ../results_sculptingstudy/sr_m${mass} --mode CATHODE --cf_separate_val_set --no_extra_signal --cf_n_samples 400000 --cf_realistic_conditional --cf_oversampling --cf_no_logit --cf_use_class_weights --cf_save_model --cf_n_runs 1 --DE_config_file ../DE_MAF_model.yml --cf_config_file ../classifier.yml
done
