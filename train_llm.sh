#!/bin/bash

helpFunction() {
    echo ""
    echo "Usage: $0 -m model_suffix [-t mode]"
    echo -e "\t-m Specify the suffix of the config file (e.g., torch, llama, hgface, test)"
    echo -e "\t-t Training mode selection (batch or iter, default: batch)"
    exit 1
}

# Default values
train_mode="batch"

while getopts "m:t:" opt; do
    case "$opt" in
        m ) model_suffix="$OPTARG" ;;
        t ) train_mode="$OPTARG" ;;
        ? ) helpFunction ;; 
    esac
done

if [ -z "$model_suffix" ]; then
    echo "Error: Missing required parameter -m"
    helpFunction
fi

# Validate mode parameter
if [[ "$train_mode" != "batch" && "$train_mode" != "iter" ]]; then
     echo "Error: -t value must be 'batch' or 'iter'."
     exit 1
fi

# Construct the PARAMS_FILE path
PARAMS_FILE="configs/${model_suffix}.yaml"

# Check if the config file exists
if [ ! -f "$PARAMS_FILE" ]; then
     echo "Error: Config file '$PARAMS_FILE' does not exist."
     exit 1
fi

# Load environment variables
source .env

# Set job-specific parameters
JOB_NAME=$model_suffix

# Submit job
sbatch --job-name="$JOB_NAME.$train_mode" \
     --account="$ACCOUNT" \
     --partition="$PARTITION" \
     --time="$TIMEOUT" \
     --nodes="$NB_NODES" \
     --ntasks-per-node="$NB_TASKS_PER_NODE" \
     --cpus-per-task="$CPUS_PER_TASK" \
     --gres="$GRES" \
     --constraint="$CONSTRAINT" \
     --mem="$MEM" \
     --output="$OUTPUT_DIR/$JOB_NAME.$train_mode.txt" \
     --export=ENV_DIR="$ENV_DIR",PARAMS_FILE="$PARAMS_FILE",TRAIN_MODE="$train_mode" \
     slurm/train_llm.slurm
