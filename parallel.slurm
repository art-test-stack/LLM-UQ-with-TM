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
module load Python/3.11.3-GCCcore-12.3.0
source $ENV_DIR
pip install -r requirements.txt
torchrun --nproc_per_node=${2:-4} main_llm.py