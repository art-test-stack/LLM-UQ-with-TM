#!/bin/sh
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo $TIMEOUT
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
module avail Python
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
source ${ENV_DIR}
cd ${TILDROOT}

pip install --upgrade pip 
pip install -r requirements.txt
pip install pyTsetlinMachine
python -c "import torch; print(torch.__version__, torch.version.cuda)"

nvidia-smi
pwd
nvcc --version

if [ "$HASH_BATCH_ID" = true ]; then
    python main_lctm_train.py --model=${MODEL} --binarizer=${BINARIZER} --hash_batch_id
else
    python main_lctm_train.py --model=${MODEL} --binarizer=${BINARIZER}
fi