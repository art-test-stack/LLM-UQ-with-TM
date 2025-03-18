#!/bin/sh
#SBATCH --export=ALL
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
# module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module avail Python
module load Python/3.11.3-GCCcore-12.3.0
source ${ENV_DIR}
# module avail Python
# module load Python/3.11.3-GCCcore-12.3.0
# source ${ENV_DIR}
source ${PIP_DIR}
# Uncomment the following lines to install requirements
pip install --upgrade pip 
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\
    python main_llm_train.py --save-model --verbose --params_file=${PARAMS_FILE}
# python main_llm_train.py --save-model --verbose --params_file=${PARAMS_FILE}
