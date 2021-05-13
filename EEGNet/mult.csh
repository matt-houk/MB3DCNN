#!/bin/csh

#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -W 240
#BSUB -q gpu
#BSUB -R "select[gtx1080]" 
#BSUB -gpu "num=1:mode=shared:mps=yes" 
#BSUB -o ./jobout/out-mult.%J
#BSUB -e ./jobout/err-mult.%J

module load conda
conda activate /usr/local/usrapps/multibranch/mjhouk/env_tensorflow
module load cuda/11.0

python runMB3D.py
