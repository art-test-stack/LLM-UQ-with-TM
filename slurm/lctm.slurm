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
# source ${PIP_DIR}
# Uncomment the following lines to install requirements

pip install --upgrade pip 
pip install -r requirements.txt
pip install pyTsetlinMachine
python -c "import torch; print(torch.__version__, torch.version.cuda)"

nvidia-smi
pwd
nvcc --version
# export PATH=/usr/local/cuda-12.4/bin:$PATH$ 
# export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH$ 
# cd $PATH
# pwd

# find / -type d -name cuda 2>/dev/null
# pip3 install pycuda

# # pip install nvidia-pyindex
# pip install pycuda
# cd pycuda_2025.1
# su -c "python distribute_setup.py"
# su -c "easy_install numpy"
# # # # sudo "python distribute_setup.py"
# # # # sudo "easy_install numpy"
# python configure.py --cuda-root=/usr/local/cuda-12.5/
# su -c "make install"
# # # pwd
# # cd ..
# cd labelcritic_tm/
# pip install .
# python examples/synthetic_lctm.py
python main_lctm_train.py --model=${MODEL} --binarizer=${BINARIZER}
# # cd ../
# # pip install nvidia-pyindex
# # pip install nvidia-cuda-nvcc
# # pip install git+https://github.com/cair/tmu.git
# # pip install pycuda
# cd examples/
# python synthetic_lctm.py
# # python lctm_eg.py