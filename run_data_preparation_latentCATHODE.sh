wget https://zenodo.org/record/4536377/files/events_anomalydetection_v2.features.h5
wget https://zenodo.org/record/5759087/files/events_anomalydetection_qcd_extra_inneronly_features.h5

python run_data_preparation_latentCATHODE.py \
    --outdir latentCATHODE_data/no-shift_bkg-only \
    --S_over_B 0

python run_data_preparation_latentCATHODE.py \
    --outdir latentCATHODE_data/no-shift_signal

python run_data_preparation_latentCATHODE.py \
    --outdir latentCATHODE_data/shifted_bkg-only \
    --S_over_B 0 \
    --data_shift 0.1

python run_data_preparation_latentCATHODE.py \
    --outdir latentCATHODE_data/shifted_signal \
    --data_shift 0.1

python run_data_preparation_latentCATHODE.py \
    --outdir latentCATHODE_data/deltaR_bkg-only \
    --S_over_B 0 \
    --add_deltaR

python run_data_preparation_latentCATHODE.py \
    --outdir latentCATHODE_data/deltaR_signal \
    --add_deltaR