#! /bin/bash
#SBATCH --account carter-gpu
#SBATCH --partition carter-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --error ./err/%x.%A.err
#SBATCH --output ./out/%x.%A.out

date
echo -e "Job ID: $SLURM_JOB_ID"

source activate /cellar/users/aklie/opt/miniconda3/envs/pytorch_dev2
echo -e "python ../drpredict/MultiEncoder.py fit --seed_everything 13 --config $1"
python ../drpredict/MultiEncoder.py fit --config $1

date
