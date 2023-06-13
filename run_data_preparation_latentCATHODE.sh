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