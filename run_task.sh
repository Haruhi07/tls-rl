#!/bin/sh

# Job name
#PBS -N tls_rl

# Output file
#PBS -o tls_rl_output.log

# Error file
#PBS -e tls_rl_err.log

# request resources and set limits
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=8:ngpus=4:mem=16GB
#:ompthreads=24
# 'select' chooses number of nodes.

#  load required modules
module load lang/python/anaconda/ lang/cuda
source activate tls_rl

# We might need to add the global paths to our code to the pythonpath. Also set the data directories globally.
cd /home/hs20307/tls-rl
export PYTHONPATH=$PYTHONPATH:"/home/hs20307/tls-rl/"

#  run the script
export DATASET=./dataset/t1
export PERL5LIB="/home/hs20307/perl5/lib/perl5"

python -u clust.py --dataset $DATASET

# To submit: qsub run_task.sh
# To display the queue: qstat -Q gpu (this is usually where the GPU job ends up)
# Display server status: qstat -B <server>
# Display job information: qstat <jobID>

# To monitor job progress:
# qstat -f | grep exec_host
# Find the node where this job is running.
# ssh to the node.
# tail /var/spool/pbs/spool/<job ID>.bp1.OU
