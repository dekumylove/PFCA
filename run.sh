################################ Section 1: Best AUC and F1 for PFCA ###############################
############# MIMIC-III dataset ######################
python train.py --model PFCA --input_dim 2850 --hidden_dim 256 --output_dim 90 --lambda 0.5 --K 3 --data_type mimic-iii --dropout_ratio 0.1 --decay 0.0001 -lr 0.001
############# MIMIC-IV dataset ######################
python train.py --model PFCA --input_dim 1992 --hidden_dim 256 --output_dim 80 --lambda 0.5 --K 3 --data_type mimic-iv --dropout_ratio 0.1 --decay 0.0001 -lr 0.001

################################ Section 2: Best AUC and F1 for Baseline Methods #################################
#####################Please modify the input data according to the needs of different models #####################
############# Lstm ###############
python train_baseline.py --model LSTM --input_dim 2850 --hidden_dim 256 --output_dim 90 --data_type mimic-iii --only_dipole --decay 0.0001 -lr 0.001
python train_baseline.py --model LSTM --input_dim 1992 --hidden_dim 256 --output_dim 80 --data_type mimic-iv --only_dipole --decay 0.0001 -lr 0.001
############# Dipole ###############
python train_baseline.py --model Dip_g --input_dim 2850 --hidden_dim 256 --output_dim 90 --data_type mimic-iii --only_dipole --decay 0.0001 -lr 0.001
python train_baseline.py --model Dip_g --input_dim 1992 --hidden_dim 256 --output_dim 80 --data_type mimic-iv --only_dipole --decay 0.0001 -lr 0.001
############# Retain ###############
python train_baseline.py --model Retain --input_dim 2850 --hidden_dim 256 --output_dim 90 --data_type mimic-iii --only_dipole --decay 0.0001 -lr 0.001
python train_baseline.py --model Retain --input_dim 1992 --hidden_dim 256 --output_dim 80 --data_type mimic-iv --only_dipole --decay 0.0001 -lr 0.001
############# HAP ###############
python train_baseline.py --model HAP --input_dim 2850 --hidden_dim 256 --output_dim 90 --data_type mimic-iii --only_dipole --decay 0.0001 -lr 0.001
python train_baseline.py --model HAP --input_dim 1992 --hidden_dim 256 --output_dim 80 --data_type mimic-iv --only_dipole --decay 0.0001 -lr 0.001
############# GraphCare ###############
python train_baseline.py --model GraphCare --input_dim 2850 --hidden_dim 256 --output_dim 90 --data_type mimic-iii --dropout_ratio 0.2 --gamma_GraphCare 0.1 --decay 0.0001 -lr 0.005
python train_baseline.py --model GraphCare --input_dim 1992 --hidden_dim 256 --output_dim 80 --data_type mimic-iv --dropout_ratio 0.2 --gamma_GraphCare 0.1 --decay 0.0001 -lr 0.005
############# MedPath ###############
python train_baseline.py --model MedPath --input_dim 2850 --hidden_dim 256 --output_dim 90 --K 2 --data_type mimic-iii --dropout_ratio 0.2 --alpha_MedPath 0.2 --decay 0.0001 -lr 0.005
python train_baseline.py --model MedPath --input_dim 1992 --hidden_dim 256 --output_dim 80 --K 3 --data_type mimic-iv --dropout_ratio 0.2 --alpha_MedPath 0.2 --decay 0.0001 -lr 0.005
############# HAR ###############
python train_baseline.py --model StageAware --input_dim 2850 --hidden_dim 256 --output_dim 90 --data_type mimic-iii --lambda_HAR 0.1 --dropout_ratio 0.2 --decay 0.0001 v
python train_baseline.py --model StageAware --input_dim 1992 --hidden_dim 256 --output_dim 80 --data_type mimic-iv --lambda_HAR 0.1 --dropout_ratio 0.2 --decay 0.0001 -lr 0.005

############################### Section 3: Interpretation ####################################
############# MIMIC-III dataset ######################
python train.py --model PFCA --input_dim 2850 --hidden_dim 256 --output_dim 90 --lambda 0.5 --K 3 --data_type mimic-iii --dropout_ratio 0.1 --show_interpretation --decay 0.0001 -lr 0.001
############# MIMIC-IV dataset ######################
python train.py --model PFCA --input_dim 1992 --hidden_dim 256 --output_dim 80 --lambda 0.5 --K 3 --data_type mimic-iv --dropout_ratio 0.1 --show_interpretation --decay 0.0001 -lr 0.001
