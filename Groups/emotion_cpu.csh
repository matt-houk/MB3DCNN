#!/bin/csh

#BSUB -J job_arr[1]
#BSUB -x
#BSUB -R "span[hosts=1]"
#BSUB -W 900
#BSUB -o ./out/trial-%I-%J.out
#BSUB -e ./out/trial-%I-%J.err

set gpu = "`nvidia-smi --query-gpu=gpu_name --format=csv,noheader`"
echo "Found $gpu GPU"
if ( "$gpu" =~ *"K20m"* ) then
	set cuda_cmd = "module load cuda/10.1"
	set env_cmd = "conda activate tf220_py377"
else if ( "$gpu" =~ *"P100"* ) then
	set cuda_cmd = "module load cuda/11.0"
	set env_cmd = "conda activate tf241_py377"
else if ( "$gpu" =~ *"GTX 1080"* ) then
	set cuda_cmd = "module load cuda/11.0"
	set env_cmd = "conda activate tf241_py377"
else if ( "$gpu" =~ *"RTX 2080"* ) then
	set cuda_cmd = "module load cuda/11.0"
	set env_cmd = "conda activate tf241_py377"
else
	echo "Running trial without GPU using CPU" 
	set cuda_cmd = "module load cuda/11.0"
	set env_cmd = "conda activate tf241_py377"
endif

echo "Loading libraries"

eval $env_cmd
eval $cuda_cmd

setenv OMP_NUM_THREADS 2

echo "Running OAT Trial $LSB_JOBINDEX with ID $LSB_JOBID"
python runEmotion.py ${LSB_JOBINDEX} ${LSB_JOBID}
