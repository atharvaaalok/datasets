#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name shapeopt_rans
#SBATCH --partition jjalonso
#SBATCH --output log/%x_%A_%a.out
#SBATCH --array=0-99


uv run simulation_multiprocessing_tutorial.py --airfoil_idx $SLURM_ARRAY_TASK_ID