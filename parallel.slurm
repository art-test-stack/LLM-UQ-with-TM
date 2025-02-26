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
source ${ENV_DIR}
source ${PIP_DIR}
# Uncomment the following lines to install requirements
# pip install --upgrade pip 
# pip install -r requirements.txt
echo ${BATCH_SIZE}
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python main_fsdp.py  --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} --lr ${LEARNING_RATE} \
        --gamma ${GAMMA} --seed ${SEED} --save-model --patience ${PATIENCE} --min-delta ${MIN_DELTA} \
        --model-dir ${MODEL_DIR} --verbose
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#     torchrun main_fsdp.py  --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} --lr ${LEARNING_RATE} \
#         --gamma ${GAMMA} --seed ${SEED} --save-model --patience ${PATIENCE} --min-delta ${MIN_DELTA} \
#         --model-dir ${MODEL_DIR}