#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=debug
#SBATCH --job-name=stimOzCz
#SBATCH --ntasks=10
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=F_r_e@hotmail.es
#SBATCH --error=logs/err-%j.log
#SBATCH --output=logs/out-%j.log
##------------------------ End job description ------------------------

module purge && module load Python

srun python stimOzCz_main.py


