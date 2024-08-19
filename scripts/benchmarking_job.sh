#!/bin/sh
# Grid Engine options (lines prefixed with #$)

# Name of job
#$ -N j_leadbetter_benchmarking
# Directory in which to run code (-cwd or -wd <path-to-wd>)
#$ -wd /home/s2603968/julia-baroclinic
# Requested runtime allowance
#$ -l h_rt=24:00:00
# Requested memory (per core)
#$ -l h_vmem=32G
# Requested number of cores in parallel environment
#$ -pe sharedmem 1
# Email address for notifications
#$ -M s2603968@ed.ac.uk
# Option to request resource reservation
#$ -R y
# Where to pipe the python output to.
#$ -o benchmarking.out
#$ -e benchmarking.err

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load conda
module load anaconda

# Load Julia
module load roslin/julia/1.9.0

# Activate conda environment
conda activate diss4

# Run the python benchmarking
python -u src/pyqg/benchmarking.py

# Run the Julia benchmarking
julia src/benchmarking.jl
