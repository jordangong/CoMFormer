#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16GB:ngpus=2
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N ADE-S50
#PBS -q gpu

module load anaconda/3.2020.2
module load devtoolset-7/gcc-7.3.1
module load openmpi/4.0.5/gcc7-ib
source activate /home/fcermelli/.conda/envs/m2f/
cd /work/fcermelli/fcdl/Mask2Former

port=$(python get_free_port.py)

cfg_file=configs/ade20k/semantic-segmentation/maskformer2_R101_bs16_90k.yaml
base=ade_ss
# MODEL.MASK_FORMER.SOFTMASK True MODEL.MASK_FORMER.DICE_WEIGHT 1. MODEL.MASK_FORMER.CLASS_WEIGHT 1. MODEL.MASK_FORMER.MASK_WEIGHT 0.0 MODEL.MASK_FORMER.FOCAL True
cont_args="CONT.BASE_CLS 50 CONT.INC_CLS 50 CONT.MODE overlap SEED 42"
task=mya_50-50-ov

#name=PerPixel
#meth_args="MODEL.MASK_FORMER.TEST.MASK_BG True MODEL.MASK_FORMER.PER_PIXEL True"
#name=MF
#meth_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False"
name=MxF
meth_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False MODEL.MASK_FORMER.SOFTMASK True MODEL.MASK_FORMER.FOCAL True"

#meth_args="${meth_args} MODEL.MASK_FORMER.NUM_OBJECT_QUERIES ${nq}"
comm_args="--dist-url tcp://127.0.0.1:${port} OUTPUT_DIR ${base} ${meth_args} ${cont_args}"
inc_args="CONT.TASK 1 CONT.WEIGHTS ${base}/${task}/${name}/step0/model_final.pth SOLVER.MAX_ITER 20000 SOLVER.BASE_LR 0.00005"

#python train_inc.py --resume --num-gpus 2 --config-file ${cfg_file} ${comm_args} CONT.TASK 0 NAME ${name} SOLVER.MAX_ITER 80000

#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_t08 CONT.DIST.PSEUDO True
python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_t08_POD CONT.DIST.PSEUDO True CONT.DIST.POD_WEIGHT 0.0001
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_MIB_200 CONT.DIST.KD_WEIGHT 200. CONT.DIST.UKD True CONT.DIST.UCE True

inc_args="CONT.TASK 2 SOLVER.MAX_ITER 20000 SOLVER.BASE_LR 0.00005"
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_T2_POD001 CONT.DIST.PSEUDO True CONT.DIST.PSEUDO_TYPE 1  CONT.DIST.POD_WEIGHT 0.001
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_MIB_200 CONT.DIST.KD_WEIGHT 200. CONT.DIST.UKD True CONT.DIST.UCE True
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_t08 CONT.DIST.PSEUDO True
python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_t08_POD CONT.DIST.PSEUDO True CONT.DIST.POD_WEIGHT 0.0001

