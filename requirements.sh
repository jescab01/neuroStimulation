#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=standard
#SBATCH --job-name=req
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=00:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=F_r_e@hotmail.es
#SBATCH --error=logs/err-%j.log
#SBATCH --output=logs/out-%j.log
##------------------------ End job description ------------------------

module load Python

srun pip install -r requirements.txt