#!/bin/bash 
#SBATCH --job-name=exp_srdiff_test_inference_joint_srdiff_lcc_maxvit_hr5_convformer_sr4_caf_focal_all_eolab
#SBATCH --partition=tnt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G
#SBATCH --time=48:00:00
#SBATCH --mail-user=kanyamahanga@ipi.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output logs/exp_srdiff_test_inference_joint_srdiff_lcc_maxvit_hr5_convformer_sr4_caf_focal_all_eolab_%j.out
#SBATCH --error logs/exp_srdiff_test_inference_joint_srdiff_lcc_maxvit_hr5_convformer_sr4_caf_focal_all_eolab_%j.err
source load_modules.sh
export CONDA_ENVS_PATH=$HOME/.conda/envs
export DATA_DIR=$BIGWORK
conda activate flair_venv
which python
cd $HOME/MISR_JOINT_SRDiff_LCC_MaxViT_HR5_ConvFormer_SR4_CAF_FOCAL_ALL_EOLAB
srun python trainer.py --config configs/diffsr_highresnet_ltae.yaml --config_file flair-config-server.yml --exp_name misr/srdiff_highresnet_ltae_ckpt --hparams="cond_net_ckpt=/bigwork/nhgnkany/Results/MISR_JOINT_SRDiff_LCC_MaxViT_HR5_ConvFormer_SR4_CAF_FOCAL_ALL_EOLAB/results/checkpoints/misr/srdiff_highresnet_ltae_ckpt" --infer
