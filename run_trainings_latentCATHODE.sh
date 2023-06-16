
# Snippets to run the trainings that reproduce the results from the LaCATHODE paper.
# The training script calls are presented sequentially, but should ideally be run in parallel on a cluster.

# needs to be on the LaCATHODE branch, whose CATHODE mode is running the classifier in the latent space
git checkout LaCATHODE

# no shift, bkg-only, latentCATHODE
for i in {1..10}; do
    python run_all.py --data_dir latentCATHODE_data/no-shift_bkg-only/ --mode CATHODE --cf_separate_val_set --no_extra_signal --save_dir latentCATHODE_trainings/latentCATHODE_bkg-only_${i} --cf_n_samples 267000 --cf_realistic_conditional --cf_oversampling --cf_no_logit --cf_use_class_weights --cf_save_model --cf_n_runs 1 --DE_config_file DE_MAF_model.yml
done

# no shift, signal, latentCATHODE
for i in {1..10}; do
    python run_all.py --data_dir latentCATHODE_data/no-shift_signal/ --mode CATHODE --cf_separate_val_set --no_extra_signal --save_dir latentCATHODE_trainings/latentCATHODE_signal_${i} --cf_n_samples 267000 --cf_realistic_conditional --cf_oversampling --cf_no_logit --cf_use_class_weights --cf_save_model --cf_n_runs 1 --DE_config_file DE_MAF_model.yml
done

# shifted, bkg-only, latentCATHODE
for i in {1..10}; do
    python run_all.py --data_dir latentCATHODE_data/shifted_bkg-only/ --mode CATHODE --cf_separate_val_set --no_extra_signal --save_dir latentCATHODE_trainings/latentCATHODE_bkg-only_shifted_${i} --cf_n_samples 267000 --cf_realistic_conditional --cf_oversampling --cf_no_logit --cf_use_class_weights --cf_save_model --cf_n_runs 1 --DE_config_file DE_MAF_model.yml
done

## shifted, signal, latentCATHODE
for i in {1..10}; do
    python run_all.py --data_dir latentCATHODE_data/shifted_signal/ --mode CATHODE --cf_separate_val_set --no_extra_signal --save_dir latentCATHODE_trainings/latentCATHODE_signal_shifted_${i} --cf_n_samples 267000 --cf_realistic_conditional --cf_oversampling --cf_no_logit --cf_use_class_weights --cf_save_model --cf_n_runs 1 --DE_config_file DE_MAF_model.yml
done

# deltaR, bkg-only, latentCATHODE
for i in {1..10}; do
    python run_all.py --data_dir latentCATHODE_data/deltaR_bkg-only/ --mode CATHODE --cf_separate_val_set --no_extra_signal --save_dir latentCATHODE_trainings/latentCATHODE_bkg-only_deltaR_${i} --cf_n_samples 267000 --cf_realistic_conditional --cf_oversampling --cf_no_logit --cf_use_class_weights --cf_save_model --cf_n_runs 1 --DE_config_file DE_MAF_model_deltaR.yml
done

# deltaR, signal, latentCATHODE
for i in {1..10}; do
    python run_all.py --data_dir latentCATHODE_data/deltaR_signal/ --mode CATHODE --cf_separate_val_set --no_extra_signal --save_dir latentCATHODE_trainings/latentCATHODE_signal_deltaR_${i} --cf_n_samples 267000 --cf_realistic_conditional --cf_oversampling --cf_no_logit --cf_use_class_weights --cf_save_model --cf_n_runs 1 --DE_config_file DE_MAF_model_deltaR.yml
done


# The idealized anomaly detector is not as well maintained in the LaCATHODE branch, so we switch to the main CATHODE paper branch.
git checkout main

# no shift, bkg-only, idealizedAD
python run_all.py --data_dir latentCATHODE_data/no-shift_bkg-only/ --mode idealized_AD --cf_separate_val_set --no_extra_signal --cf_no_logit --cf_oversampling --cf_use_class_weights --cf_save_model --cf_extra_bkg --save_dir latentCATHODE_trainings/idealizedAD_bkg-only --cf_n_runs 10

# no shift, signal, idealizedAD
python run_all.py --data_dir latentCATHODE_data/no-shift_signal/ --mode idealized_AD --cf_separate_val_set --no_extra_signal --cf_no_logit --cf_oversampling --cf_use_class_weights --cf_save_model --cf_extra_bkg --save_dir latentCATHODE_trainings/idealizedAD_signal --cf_n_runs 10

# shifted, bkg-only, idealizedAD
python run_all.py --data_dir latentCATHODE_data/shifted_bkg-only/ --mode idealized_AD --cf_separate_val_set --no_extra_signal --cf_no_logit --cf_oversampling --cf_use_class_weights --cf_save_model --cf_extra_bkg --save_dir latentCATHODE_trainings/idealizedAD_bkg-only_shifted --cf_n_runs 10

# shifted, signal, idealizedAD
python run_all.py --data_dir latentCATHODE_data/shifted_signal/ --mode idealized_AD --cf_separate_val_set --no_extra_signal --cf_no_logit --cf_oversampling --cf_use_class_weights --cf_save_model --cf_extra_bkg --save_dir latentCATHODE_trainings/idealizedAD_signal_shifted --cf_n_runs 10

# deltaR, bkg-only, idealizedAD
python run_all.py --data_dir latentCATHODE_data/deltaR_bkg-only/ --mode idealized_AD --cf_separate_val_set --no_extra_signal --cf_no_logit --cf_oversampling --cf_use_class_weights --cf_save_model --cf_extra_bkg --save_dir latentCATHODE_trainings/idealizedAD_bkg-only_deltaR --cf_n_runs 10

# deltaR, signal, idealizedAD
python run_all.py --data_dir latentCATHODE_data/deltaR_signal/ --mode idealized_AD --cf_separate_val_set --no_extra_signal --cf_no_logit --cf_oversampling --cf_use_class_weights --cf_save_model --cf_extra_bkg --save_dir latentCATHODE_trainings/idealizedAD_signal_deltaR --cf_n_runs 10


# Also for the classic CATHODE implementation we rely on the main branch.
git checkout main

# But we don't redo the density estimation. We rather copy over the DE trainings from the LaCATHODE runs. In addition to saving computational cost, this ensures that the DE has been trained on SB data with a full train/val/test separation, which was not necessary in the original CATHODE study but is needed for an unbiased background sculpting study.
for i in {1..10}; do
    mkdir latentCATHODE_trainings/classicCATHODE_bkg-only_${i}
    cp latentCATHODE_trainings/latentCATHODE_bkg-only_${i}/my_ANODE_model* latentCATHODE_trainings/classicCATHODE_bkg-only_${i}/
    
    mkdir latentCATHODE_trainings/classicCATHODE_signal_${i}
    cp latentCATHODE_trainings/latentCATHODE_signal_${i}/my_ANODE_model* latentCATHODE_trainings/classicCATHODE_signal_${i}/

    mkdir latentCATHODE_trainings/classicCATHODE_bkg-only_shifted_${i}
    cp latentCATHODE_trainings/latentCATHODE_bkg-only_shifted_${i}/my_ANODE_model* latentCATHODE_trainings/classicCATHODE_bkg-only_shifted_${i}/

    mkdir latentCATHODE_trainings/classicCATHODE_signal_shifted_${i}
    cp latentCATHODE_trainings/latentCATHODE_signal_shifted_${i}/my_ANODE_model* latentCATHODE_trainings/classicCATHODE_signal_shifted_${i}/

    mkdir latentCATHODE_trainings/classicCATHODE_bkg-only_deltaR_${i}
    cp latentCATHODE_trainings/latentCATHODE_bkg-only_deltaR_${i}/my_ANODE_model* latentCATHODE_trainings/classicCATHODE_bkg-only_deltaR_${i}/

    mkdir latentCATHODE_trainings/classicCATHODE_signal_deltaR_${i}
    cp latentCATHODE_trainings/latentCATHODE_signal_deltaR_${i}/my_ANODE_model* latentCATHODE_trainings/classicCATHODE_signal_deltaR_${i}/
done

# Also, the DE_MAF_model_deltaR.yml config file doesn't exist on the main branch. Let's quickly create it.
sed "s/num_inputs: 4/num_inputs: 5/" DE_MAF_model.yml > DE_MAF_model_deltaR.yml

# no shift, bkg-only, classicCATHODE
for i in {1..10}; do
    python run_all.py --data_dir latentCATHODE_data/no-shift_bkg-only/ --mode CATHODE --cf_separate_val_set --no_extra_signal --save_dir latentCATHODE_trainings/classicCATHODE_bkg-only_${i} --cf_n_samples 267000 --cf_realistic_conditional --cf_oversampling --cf_no_logit --cf_use_class_weights --cf_save_model --cf_n_runs 1 --DE_config_file DE_MAF_model.yml --DE_skip
done

# no shift, signal, classicCATHODE
for i in {1..10}; do
    python run_all.py --data_dir latentCATHODE_data/no-shift_signal/ --mode CATHODE --cf_separate_val_set --no_extra_signal --save_dir latentCATHODE_trainings/classicCATHODE_signal_${i} --cf_n_samples 267000 --cf_realistic_conditional --cf_oversampling --cf_no_logit --cf_use_class_weights --cf_save_model --cf_n_runs 1 --DE_config_file DE_MAF_model.yml --DE_skip
done

# shifted, bkg-only, classicCATHODE
for i in {1..10}; do
    python run_all.py --data_dir latentCATHODE_data/shifted_bkg-only/ --mode CATHODE --cf_separate_val_set --no_extra_signal --save_dir latentCATHODE_trainings/classicCATHODE_bkg-only_shifted_${i} --cf_n_samples 267000 --cf_realistic_conditional --cf_oversampling --cf_no_logit --cf_use_class_weights --cf_save_model --cf_n_runs 1 --DE_config_file DE_MAF_model.yml --DE_skip
done

# shifted, signal, classicCATHODE
for i in {1..10}; do
    python run_all.py --data_dir latentCATHODE_data/shifted_signal/ --mode CATHODE --cf_separate_val_set --no_extra_signal --save_dir latentCATHODE_trainings/classicCATHODE_signal_shifted_${i} --cf_n_samples 267000 --cf_realistic_conditional --cf_oversampling --cf_no_logit --cf_use_class_weights --cf_save_model --cf_n_runs 1 --DE_config_file DE_MAF_model.yml --DE_skip
done

# deltaR, bkg-only, classicCATHODE
for i in {1..10}; do
    python run_all.py --data_dir latentCATHODE_data/deltaR_bkg-only/ --mode CATHODE --cf_separate_val_set --no_extra_signal --save_dir latentCATHODE_trainings/classicCATHODE_bkg-only_deltaR_${i} --cf_n_samples 267000 --cf_realistic_conditional --cf_oversampling --cf_no_logit --cf_use_class_weights --cf_save_model --cf_n_runs 1 --DE_config_file DE_MAF_model_deltaR.yml --DE_skip
done

# deltaR, signal, classicCATHODE
for i in {1..10}; do
    python run_all.py --data_dir latentCATHODE_data/deltaR_signal/ --mode CATHODE --cf_separate_val_set --no_extra_signal --save_dir latentCATHODE_trainings/classicCATHODE_signal_deltaR_${i} --cf_n_samples 267000 --cf_realistic_conditional --cf_oversampling --cf_no_logit --cf_use_class_weights --cf_save_model --cf_n_runs 1 --DE_config_file DE_MAF_model_deltaR.yml --DE_skip
done


# Don't forget to switch to the LaCATHODE branch for evaluating,
git checkout LaCATHODE
