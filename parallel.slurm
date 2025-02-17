source .env 
#!/bin/sh
#SBATCH -J llm
#SBATCH --partition=GPUQ
#SBATCH --time=0-$TIMEOUT
#SBATCH --nodes=$NB_NODES
#SBATCH --ntasks-per-node=$NTASKS_PER_NODES
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --gres=$GRES
#SBATCH --mem=$MEM
#SBATCH --output=$LOGS_FILE
### SBATCH --mail-user=$EMAIL
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
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
source $DIR_ENV
pip install -r requirements.txt
torchrun --nproc_per_node=${2:-4} main_llm.py