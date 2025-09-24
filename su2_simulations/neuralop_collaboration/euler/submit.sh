#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name shapeopt_euler
#SBATCH --partition jjalonso
#SBATCH --output log/%x_%a_%A.out
#SBATCH --array=0-99


apptainer exec containers/Apptainer.sif \
    bash -lc 'uv run simulation_multiprocessing_tutorial.py --airfoil_idx $SLURM_ARRAY_TASK_ID'
