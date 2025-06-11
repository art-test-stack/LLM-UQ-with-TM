#!/bin/bash

# Default values
BINARIZER='default'
HASH_BATCH_ID='False'
DOCUMENT='accumulation'

# Parse arguments
while getopts "m:b:d:h" opt; do
    case $opt in
        m) MODEL="$OPTARG" ;;
        b) BINARIZER="$OPTARG" ;;
        d) DOCUMENT="$OPTARG" ;;
        h) HASH_BATCH_ID='True' ;;
        *) echo "Usage: $0 -m <model> [-b <binarizer>] [-d <document>] [-h]"; exit 1 ;;
    esac
done

# Check required argument
if [ -z "$MODEL" ]; then
    echo "Error: -m <model> is required."
    echo "Usage: $0 -m <model> [-b <binarizer>] [-h] [-d <document>]"
    exit 1
fi
if [ "$HASH_BATCH_ID" = "True" ]; then
    JOB_NAME="hash.${TM_JOB_NAME}.${MODEL}.${BINARIZER}.${DOCUMENT}"
else
    JOB_NAME="${TM_JOB_NAME}.${MODEL}.${BINARIZER}.${DOCUMENT}"
fi

source .env && sbatch --job-name=$JOB_NAME \
    --account=$ACCOUNT \
    --partition=$TM_PARTITION \
    --time=$TM_TIMEOUT \
    --nodes=$TM_NB_NODES \
    --ntasks-per-node=$TM_NB_TASKS_PER_NODE \
    --cpus-per-task=$TM_CPUS_PER_TASK \
    --gres=$TM_GRES \
    --constraint=$TM_CONSTRAINT \
    --mem=$TM_MEM \
    --output=$OUTPUT_DIR/$JOB_NAME.txt \
    --export=ENV_DIR=$ENV_DIR,MODEL=$MODEL,BINARIZER=$BINARIZER,HASH_BATCH_ID=$HASH_BATCH_ID,DOCUMENT=$DOCUMENT \
    slurm/lctm.slurm