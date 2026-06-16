module purge
module load TACC gcc/13.2.0 impi/21.12 cuda/12.2   # load cuda LAST
unset PYTHONPATH
source /work/11547/jlchan/ls6/apps/miniforge3/bin/activate
conda activate pyfr

# Intel MPI threading (do NOT set I_MPI_THREAD — unsupported on 21.12)
unset I_MPI_THREAD
export I_MPI_THREAD_LEVEL=3

# CUDA / NVRTC (if libnvrtc.so not found)
export PYFR_NVRTC_LIBRARY_PATH=/lib64/libnvrtc.so

# export CUDA_VISIBLE_DEVICES=0   # single-GPU only

export PYFR_NVRTC_LIBRARY_PATH=$TACC_CUDA_DIR/lib64/libnvrtc.so
export LD_LIBRARY_PATH=$TACC_CUDA_DIR/lib64:$LD_LIBRARY_PATH

# to run on 3 GPUs assuming all is set up properly above
# - partition
# pyfr partition add kh-coarse.pyfrm 3

# - check partition
# pyfr partition list kh-coarse.pyfrm
# pyfr partition info kh-coarse.pyfrm 3

# - run
# ibrun -n 3 pyfr -p run -b cuda kh-coarse.pyfrm kh.ini
