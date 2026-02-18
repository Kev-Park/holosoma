#!/bin/bash
#SBATCH --job-name=holosoma     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=24G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of GPUs per node
#SBATCH --constraint=gpu80
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=kp0374@princeton.edu

module purge
module load anaconda3/2025.6
conda activate hsretargeting

export CUDA_VISIBLE_DEVICES=0

python examples/robot_retarget.py --data_path demo_data/OMOMO_new --task-type object_interaction --task-name sub3_largebox_003 --data_format smplh --retargeter.debug --retargeter.visualize