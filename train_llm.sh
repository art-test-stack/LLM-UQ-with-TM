#!/bin/bash

helpFunction() {
   echo ""
   echo "Usage: $0 -t model_type"
   echo -e "\t-t Choose model type (torch or llama)"
   exit 1
}

while getopts "t:" opt; do
   case "$opt" in
      t ) model_type="$OPTARG" ;;
      ? ) helpFunction ;; 
   esac
done

if [ -z "$model_type" ]; then
   echo "Error: Missing required parameter -t"
   helpFunction
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
else 
    echo "Error: -t value must be 'torch' or 'llama'."
    exit 1
fi

# Submit job
sbatch --job-name="$JOB_NAME.$RUN_TYPE" \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$TIMEOUT" \
    --nodes="$NB_NODES" \
    --ntasks-per-node="$NB_TASKS_PER_NODE" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --gres="$GRES" \
    --constraint="$CONSTRAINT" \
    --mem="$MEM" \
    --output="$OUTPUT_DIR/$JOB_NAME.$RUN_TYPE.txt" \
    --export=ENV_DIR="$ENV_DIR",PARAMS_FILE="$PARAMS_FILE" \
    train_llm.slurm
