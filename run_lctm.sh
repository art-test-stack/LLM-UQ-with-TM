#!/bin/bash
MODEL=$1

source .env && sbatch --job-name=$TM_JOB_NAME.$MODEL \
    --account=$ACCOUNT \
    --partition=$TM_PARTITION \
    --time=$TM_TIMEOUT \
    --nodes=$TM_NB_NODES \
    --ntasks-per-node=$TM_NB_TASKS_PER_NODE \
    --cpus-per-task=$TM_CPUS_PER_TASK \
    --gres=$TM_GRES \
    --constraint=$TM_CONSTRAINT \
    --mem=$TM_MEM \
    --output=$OUTPUT_DIR/$TM_JOB_NAME.$MODEL.txt \
    --export=ENV_DIR=$ENV_DIR,MODEL=$MODEL \
    slurm/lctm.slurm