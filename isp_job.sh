#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=512MB
#SBATCH --output=ISP_HPC_Output.txt

#load modules
module load python/3.11.5 scipy-stack

#create and activate virtual environment
virtualenv --no-download $SLURM_TMPDIR/ISP_ENV
source $SLURM_TMPDIR/ISP_ENV/bin/activate

#install Packages
pip install --no-index --upgrade pip
pip install --no-index pandas numpy matplotlib statsmodels scikit_learn openpyxl

#run python script
python ARIMA_by_jurisdiction.py

