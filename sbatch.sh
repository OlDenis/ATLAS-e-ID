#!/bin/bash
#SBATCH --account=def-arguinj
#SBATCH --time=0-03:00     # time limit (DD-HH:MM)
#SBATCH --nodes=1          # number of nodes
#SBATCH --mem=128G         # memory per node (uncomment on Beluga)
#SBATCH --cpus-per-task=8  # number of CPU threads per node
#SBATCH --gres=gpu:4       # number of GPU(s) per node
#SBATCH --job-name=el-id
#SBATCH --output=outputs/log_files/%x_%A_%a.out
#SBATCH --array=0


export VAR=$SLURM_ARRAY_TASK_ID
export SCRIPT_VAR


# TRAINING ON LPS
#SIF=/opt/tmp/godin/sing_images/tf-2.1.0-gpu-py3_sing-2.6.sif
#singularity shell --nv --bind /lcg,/opt $SIF classifier.sh $VAR $SCRIPT_VAR
#singularity shell      --bind /lcg,/opt $SIF presampler.sh


# TRAINING ON BELUGA
SIF=/scratch/odenis/tf-2.1.0-gpu-py3_sing-3.5.sif
module load singularity/3.5
singularity shell --nv --bind /scratch/odenis $SIF < classifier.sh $VAR $SCRIPT_VAR
