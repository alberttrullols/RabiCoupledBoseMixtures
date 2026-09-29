#!/bin/bash
#SBATCH --job-name=dmc_N80_g0.0100_t0.0005_g12_0.0111
#SBATCH --chdir=.
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=24:00:00
#SBATCH --qos=gp_resc
#SBATCH -A upc59

module load julia
julia -t $SLURM_CPUS_PER_TASK $HOME/run_dmc_cc_forward_correfoc.jl 2>&1 | tee run.log
