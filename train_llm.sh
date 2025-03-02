source .env && sbatch --job-name=$JOB_NAME.$RUN_TYPE \
    --account=$ACCOUNT \
    --partition=$PARTITION \
    --time=$TIMEOUT \
    --nodes=$NB_NODES \
    --ntasks-per-node=$NB_TASKS_PER_NODE \
    --cpus-per-task=$CPUS_PER_TASK \
    --gres=$GRES \
    --constraint=$CONSTRAINT \
    --mem=$MEM \
    --output=$OUTPUT_DIR/$JOB_NAME.$RUN_TYPE.txt \
    --export=ENV_DIR=$ENV_DIR \
    parallel.slurm 