#!/bin/bash

helpFunction() {
   echo ""
   echo "Usage: $0 -m model_type [-t mode]"
   echo -e "\t-m Choose model type (torch, llama, hgface or test)"
   echo -e "\t-t Training mode selection (batch or iter, default: batch)"
   exit 1
}

# Default values
mode="batch"

while getopts "m:t:" opt; do
   case "$opt" in
      m ) model_type="$OPTARG" ;;
      t ) train_mode="$OPTARG" ;;
      ? ) helpFunction ;; 
   esac
done

if [ -z "$model_type" ]; then
   echo "Error: Missing required parameter -m"
   helpFunction
fi

# Validate mode parameter
if [[ "$train_mode" != "batch" && "$train_mode" != "iter" ]]; then
    echo "Error: -t value must be 'batch' or 'iter'."
    exit 1
fi

# Load environment variables
source .env

# Set job-specific parameters
if [ "$model_type" = "torch" ]; then 
    JOB_NAME=$JOB_NAME_TORCH
    PARAMS_FILE=$PARAMS_FILE_TORCH
elif [ "$model_type" = "llama" ]; then 
    JOB_NAME=$JOB_NAME_LLAMA
    PARAMS_FILE=$PARAMS_FILE_LLAMA
elif [ "$model_type" = "test" ]; then 
    JOB_NAME=$JOB_NAME_TEST
    PARAMS_FILE=$PARAMS_FILE_TEST
elif [ "$model_type" = "hgface" ]; then 
    JOB_NAME=$JOB_NAME_HGFACE
    PARAMS_FILE=$PARAMS_FILE_HGFACE
else 
    echo "Error: -m value must be 'torch', 'llama', 'hgface', or 'test'."
    exit 1
fi

# Submit job
sbatch --job-name="$JOB_NAME.$train_mode.$RUN_TYPE" \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$TIMEOUT" \
    --nodes="$NB_NODES" \
    --ntasks-per-node="$NB_TASKS_PER_NODE" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --gres="$GRES" \
    --constraint="$CONSTRAINT" \
    --mem="$MEM" \
    --output="$OUTPUT_DIR/$JOB_NAME.$train_mode.$RUN_TYPE.txt" \
    --export=ENV_DIR="$ENV_DIR",PARAMS_FILE="$PARAMS_FILE",TRAIN_MODE="$train_mode" \
    train_llm.slurm