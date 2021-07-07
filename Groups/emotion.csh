#!/bin/csh

#BSUB -J job_arr[8]
#BSUB -x
#BSUB -R "span[hosts=1]"
#BSUB -W 60
#BSUB -q gpu
#BSUB -R "select[p100]"
#BSUB -gpu "num=1:mode=shared:mps=yes"
#BSUB -o ./out/trial-%I-%J.out
#BSUB -e ./out/trial-%I-%J.err

set run = true
set gpu = "`nvidia-smi --query-gpu=gpu_name --format=csv,noheader`"
echo "Found $gpu GPU"
if ( "$gpu" =~ *"K20m"* ) then
	set cuda_cmd = "module load cuda/10.1"
	set env_cmd = "conda activate /usr/local/usrapps/multibranch/mjhouk/tf220_py377"
else if ( "$gpu" =~ *"P100"* ) then
	set cuda_cmd = "module load cuda/11.0"
	set env_cmd = "conda activate /usr/local/usrapps/multibranch/mjhouk/tf241_py377"
else if ( "$gpu" =~ *"GTX 1080"* ) then
	set cuda_cmd = "module load cuda/11.0"
	set env_cmd = "conda activate /usr/local/usrapps/multibranch/mjhouk/tf241_py377"
else if ( "$gpu" =~ *"RTX 2080"* ) then
	set cuda_cmd = "module load cuda/11.0"
	set env_cmd = "conda activate /usr/local/usrapps/multibranch/mjhouk/tf241_py377"
else
	set run = false
endif

if ( "$run" =~ true ) then
	
	echo "Loading libraries"

	eval $env_cmd
	eval $cuda_cmd

	echo "Running Job with ID $LSB_JOBID"
	
	python runEmotion.py $LSB_JOBINDEX $LSB_JOBID

else
	
	echo "GPU not available, cancelling job with ID $LSB_JOBID"

endif

