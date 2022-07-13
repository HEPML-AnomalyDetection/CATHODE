#!/bin/bash


# Scan over several signal region bins with a width of 400 GeV as in the original CATHODE paper.
# Setting S_over_B to 0 ensures that we do not inject any signal, as this is a background-only sculpting study
mkdir ../separated_data_sculptingstudy
python run_data_preparation_sculptingstudy.py --outdir ../separated_data_sculptingstudy/sr_3200 --S_over_B 0 --seed 42 --sr_limits 3.0 3.4 --split_data_more

python run_data_preparation_sculptingstudy.py --outdir ../separated_data_sculptingstudy/sr_3400 --S_over_B 0 --seed 42 --sr_limits 3.2 3.6 --split_data_more

python run_data_preparation_sculptingstudy.py --outdir ../separated_data_sculptingstudy/sr_3600 --S_over_B 0 --seed 42 --sr_limits 3.4 3.8 --split_data_more

python run_data_preparation_sculptingstudy.py --outdir ../separated_data_sculptingstudy/sr_3800 --S_over_B 0 --seed 42 --sr_limits 3.6 4.0 --split_data_more

python run_data_preparation_sculptingstudy.py --outdir ../separated_data_sculptingstudy/sr_4000 --S_over_B 0 --seed 42 --sr_limits 3.8 4.2 --split_data_more

python run_data_preparation_sculptingstudy.py --outdir ../separated_data_sculptingstudy/sr_4200 --S_over_B 0 --seed 42 --sr_limits 4.0 4.4 --split_data_more

python run_data_preparation_sculptingstudy.py --outdir ../separated_data_sculptingstudy/sr_4400 --S_over_B 0 --seed 42 --sr_limits 4.2 4.6 --split_data_more
