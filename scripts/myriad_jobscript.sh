#!/bin/bash -l

# Request wallclock time (format hours:minutes:seconds).
#$ -l h_rt=48:00:0

# Request RAM (must be an integer followed by M, G, or T)
#$ -l mem=4G

# Request TMPDIR space (default is 10 GB - remove if cluster is diskless)
#$ -l tmpfs=1G

# Set the name of the job.
#$ -N Serial_Job

# Set the working directory to somewhere in your scratch space.
# This is a necessary step as compute nodes cannot write to $HOME.
#$ -wd /home/ucahutt/Scratch/workspace

cd $HOME/BayesianSymbolicRegression

# python3 src/main.py -c vicatsr_written_expr.json > outs/out1.txt
python3 src/main.py -c vicatsr_dso_benchmarks.json
