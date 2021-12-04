python classifier.py --host_name=beluga --n_classes=0 --n_train=0 --valid_cut='(abs(sample["eta"]) < 1.37) & (sample["pt"] > 5) & (sample["pt"] < 10)' 

exit


################################################################################################################
#### PICK A SHELL SCRIPT FROM BELOW ############################################################################
################################################################################################################


# SINGLE TRAINING
python classifier.py  --n_train=10e6  --n_valid=1e6  --batch_size=5e3  --n_epochs=100  --n_classes=2           \
                      --n_tracks=5    --l2=1e-7      --dropout=0.1     --verbose=2     --weight_type=None      \
                      --bkg_ratio=2   --output_dir=outputs  --generator=OFF  --host_name=${HOST_NAME}


# MULTIPLE TRAININGS (array jobs)
python classifier.py  --n_train=10e6      --n_valid=15e6      --batch_size=5e3   --n_epochs=100  --n_classes=2 \
                      --n_tracks=5        --l2=1e-8           --dropout=0.05     --verbose=2     --NN_type=CNN \
                      --plotting=OFF      --weight_type=None  --scalars=ON       --FCN_neurons 200 200         \
                      --output_dir=outputs --sbatch_var=${SBATCH_VAR}  --model_out=model_${SBATCH_VAR}.h5      \
                      --scaler_out=scaler_${SBATCH_VAR}.pkl   --results_out=valid_results_${SBATCH_VAR}.pkl


# CROSS-VALIDATION TRAINING (array jobs)
n_e=15e6; n_epochs=100; n_classes=2; n_tracks=5; n_folds=5; verbose=2; scalars=ON; images=ON; NN_type=CNN
fold=$SBATCH_VAR; output_dir=outputs
python classifier.py  --n_train=$n_e          --n_valid=0           --batch_size=5e3     --n_epochs=$n_epochs  \
                      --n_classes=$n_classes  --n_tracks=$n_tracks  --verbose=$verbose   --dropout=0.1         \
                      --images=$images        --scalars=$scalars    --NN_type=$NN_type   --weight_type=none    \
                      --train_cuts '(sample["eventNumber"]%'${n_folds}'!='$(($fold-1))')'                      \
                      --valid_cuts '(sample["eventNumber"]%'${n_folds}'=='$(($fold-1))')'                      \
                      --output_dir=$output_dir  --scaler_out scaler_${fold}.pkl --model_out model_${fold}.h5   \
                      --results_out=valid_results_${fold}.pkl  --generator=OFF
# POST CROSS-VALIDATION
python classifier.py  --n_train=0           --n_valid=$n_e        --n_epochs=0        --n_classes=$n_classes   \
                      --n_tracks=$n_tracks  --n_folds=${n_folds}  --NN_type=$NN_type  --images=$images         \
                      --scalars=$scalars    --output_dir=$output_dir   --valid_results_out=valid_data.pkl


# CROSS-VALIDATION TRAINING (array jobs)
n_e=15e6; n_epochs=100; n_classes=2; n_tracks=$SCRIPT_VAR; n_folds=10; verbose=2; scalars=ON; images=ON
fold=$SBATCH_VAR; output_dir=outputs/${SCRIPT_VAR}_tracks
#for ((fold = 1; fold <= $n_folds; fold++)) do
#for fold in 1 2 3 4 5 do
python classifier.py  --n_train=$n_e          --n_valid=0           --batch_size=5e3     --n_epochs=$n_epochs  \
                      --n_classes=$n_classes  --n_tracks=$n_tracks  --verbose=$verbose   --dropout=0.1         \
                      --images=$images        --scalars=$scalars    --NN_type=$NN_type   --weight_type=none    \
                      --train_cuts '(sample["eventNumber"]%'${n_folds}'!='$(($fold-1))')'                      \
                      --valid_cuts '(sample["eventNumber"]%'${n_folds}'=='$(($fold-1))')'                      \
                      --output_dir=$output_dir  --scaler_out scaler_${fold}.pkl --model_out model_${fold}.h5   \
                      --results_out=valid_results_${fold}.pkl
#done
python classifier.py  --n_train=0           --n_valid=$n_e        --n_epochs=0        --n_classes=$n_classes   \
                      --n_tracks=$n_tracks  --n_folds=${n_folds}  --NN_type=$NN_type  --images=$images         \
                      --scalars=$scalars    --output_dir=$output_dir   --results_out=valid_data.pkl


# USING VALIDATION RESULTS FOR PLOTTING
python classifier.py  --n_train=0  --n_valid=15e6  --output_dir=outputs  --results_in=valid_results.pkl        \
                      --eta_region=0.0-1.3  --plotting=OFF


# FEATURE REMOVAL IMPORTANCE RANKING (array jobs)
python classifier.py  --n_train=10e6 --n_valid=1e6 --batch_size=5e3 --n_epochs=100 --n_classes=2 --verbose=1   \
                      --weight_type=none --plotting=OFF --feature_removal=ON --sep_bkg=ON  --generator=ON      \
                      --results_out=results.pkl  --output_dir=outputs/feature_removal                          \
                      --sbatch_var=${SBATCH_VAR}  --host_name=${HOST_NAME}
