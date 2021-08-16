#!/bin/sh

# Job name
#PBS -N cluster

# Output file
#PBS -o cluster.output

# Error file
#PBS -e cluster.err

# request resources and set limits
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=8:ngpus=4:mem=16GB
#:ompthreads=24
# 'select' chooses number of nodes.

#  load required modules
module load lang/python/anaconda/ lang/cuda/11.1
source activate env

# We might need to add the global paths to our code to the pythonpath. Also set the data directories globally.
cd /work/hs20307/tls-rl
export PYTHONPATH=$PYTHONPATH:"/work/hs20307/tls-rl"

#  run the script
export DATASET=./dataset/t17
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
