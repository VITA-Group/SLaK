#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus=4
#SBATCH -t 5-00:00:00
#SBATCH --cpus-per-task=18
#SBATCH -o ./Slak_dis_120_T20_new_bntrue_ST_RN50_NKD.out

source /home/sliu/miniconda3/etc/profile.d/conda.sh
#source activate slak

cd ..

outdir=/gpfs/work3/0/prjste21060/projects/datasets/T20_bnTrue_STRN50_NKD_Test
python -m torch.distributed.launch --nproc_per_node=2 main_KD.py  \
--resume ../checkpoints/SLaK_tiny_checkpoint.pth --Decom True --T 3.0 --width_factor 1.3 -u 2000 --epochs 120 --model SLaK_tiny --model_s resnet50  --distill_type KD --drop_path 0.1 --batch_size 64 --lr 4e-3 --update_freq 8 --model_ema true --model_ema_eval false \
--data_path /scratch-shared/sliu/imagenet/ --num_workers 40 \
--kernel_size 51 49 47 13 5 --output_dir $outdir


source deactivate
