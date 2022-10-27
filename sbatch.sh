#!/bin/bash

#---------------------------------------------------------------------
# SLURM OPTIONS (LPS or BELUGA)
#---------------------------------------------------------------------
#SBATCH --account=def-arguinj
#SBATCH --time=06-00:00         #time limit (DD-HH:MM)
#SBATCH --nodes=1               #number of nodes
##SBATCH --mem=128G              #memory per node (on Beluga)
#SBATCH --cpus-per-task=4       #number of CPU threads per node
#SBATCH --gres=gpu:1            #number of GPU(s) per node
#SBATCH --job-name=e-ID_CNN
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0
#---------------------------------------------------------------------

convertsecs() {
     ((h=${1}/3600))
     ((m=(${1}%3600)/60))
     ((s=${1}%60))
     printf "%02d:%02d:%02d\n" $h $m $s
    }

START="$( date +%s )"
echo Processing job $SLURM_JOB_ID
date 

export SBATCH_VAR=$SLURM_ARRAY_TASK_ID
export  HOST_NAME=$SLURM_SUBMIT_HOST
export   NODE_DIR=$SLURM_TMPDIR
export SCRIPT_VAR

if [[ $HOST_NAME == *atlas* ]]
then
    # TRAINING ON LPS
    if   [[ -d "/nvme1" ]]
    then
	PATHS=/lcg,/opt,/nvme1
    else
	PATHS=/lcg,/opt
    fi
    SIF=/opt/tmp/godin/sing_images/tf-2.1.0-gpu-py3_sing-2.6.sif
    #singularity shell      --bind $PATHS $SIF presampler.sh
    singularity shell --nv --bind $PATHS $SIF classifier.sh $SBATCH_VAR $HOST_NAME $SCRIPT_VAR
else
    # TRAINING ON BELUGA
    if [[ -n "$NODE_DIR" ]]
    then
	echo "COPYING DATA FILES TO LOCAL NODE"
	#cp -r /project/def-arguinj/shared/e-ID_data/{0.0-1.3,1.3-1.6,1.6-2.5,0.0-2.5} $NODE_DIR
	cp -r /scratch/odenis/JF17 $NODE_DIR
    fi
    module load singularity/3.8
    PATHS=/project/def-arguinj,/scratch/odenis,$NODE_DIR,/lustre03/project/6003740/odenis/SampleProd/
    SIF=/project/def-arguinj/shared/sing_images/tf-2.1.0-gpu-py3_sing-3.5.sif
    #singularity shell --nv --bind $PATHS $SIF < classifier.sh $SBATCH_VAR $HOST_NAME $NODE_DIR
    singularity shell      --bind $PATHS $SIF < presampler.sh
fi

DURATION=$[ $( date +%s ) - $START ]
echo Duration: $( convertsecs $DURATION )
