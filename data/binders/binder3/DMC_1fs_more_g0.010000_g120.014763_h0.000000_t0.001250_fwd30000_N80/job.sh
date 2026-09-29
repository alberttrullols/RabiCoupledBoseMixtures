#!/bin/bash
#SBATCH --job-name=dmc_g12_crit_N80_g120.014763_h0.000000_t0.001250_fwd30000
#SBATCH --chdir=.
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --partition=high-cpu
#SBATCH --account=lab_upc59
#SBATCH --qos=res_upc59_a

module load modulepath/EESSI/2025.06
module load Julia/1.12.2
julia --project=~ -t $SLURM_CPUS_PER_TASK ../run_dmc_cc_forward_correfoc.jl 2>&1 | tee run.log
