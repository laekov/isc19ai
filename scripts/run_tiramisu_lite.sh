#!/bin/bash

module load tensorflow/gpu-1.13.1-py36

#openmp stuff
export OMP_NUM_THREADS=16
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread

#pick GPU
export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
# export CUDA_VISIBLE_DEVICES=$OMP_

#directories and files
datadir=/mnt/data
checkpt=/mnt/ssd/laekov/checkpt_ds_modelt
scratchdir=/mnt/data/segm_h5_v3_new_split/
numfiles_train=3000
numfiles_validation=300
numfiles_test=500

case $OMPI_COMM_WORLD_LOCAL_RANK in
	0)
		socket=0
		;;
	1 | 2)
		socket=1,2
		;;
	3)
		socket=3
		;;
esac

#network parameters
downsampling=4
batch=12
# blocks="2 2 2 4 5"
blocks="3 3 4 4 7 7"

#create run dir
run_dir=/mnt/ssd/laekov/tiramisu_run
#rundir=${WORK}/data/tiramisu/runs/run_nnodes16_j6415751
mkdir -p ${run_dir}

#copy relevant files
cp ../utils/graph_flops.py ${run_dir}/
cp ../utils/common_helpers.py ${run_dir}/
cp ../utils/data_helpers.py ${run_dir}/
cp ../tiramisu-tf/tiramisu-tf-train.py ${run_dir}/
cp ../tiramisu-tf/tiramisu-tf-inference.py ${run_dir}/
cp ../tiramisu-tf/tiramisu_model.py ${run_dir}/


#step in
cd ${run_dir}

#some parameters
lag=0
train=0
test=0
predict=1

if [ ${train} -eq 1 ]; then
  echo "Starting Training bs = " $batch " socket = " $socket
  runid=0
  runfiles=$(ls -latr out.lite.fp32.lag${lag}.train.run* | tail -n1 | awk '{print $9}')
  if [ ! -z ${runfiles} ]; then
      runid=$(echo ${runfiles} | awk '{split($1,a,"run"); print a[1]+1}')
  fi
    
  numactl --cpunodebind=$socket -m $socket \
  python -u ./tiramisu-tf-train.py      --datadir_train ${scratchdir}/train \
                                        --train_size ${numfiles_train} \
                                        --datadir_validation ${scratchdir}/validation \
                                        --validation_size ${numfiles_validation} \
                                        --chkpt_dir $checkpt \
										--disable_imsave \
										--downsampling ${downsampling} \
                                        --downsampling_mode "center-crop" \
                                        --epochs 5000 \
                                        --fs global \
                                        --blocks ${blocks} \
                                        --growth 32 \
                                        --filter-sz 5 \
                                        --loss weighted \
                                        --optimizer opt_type=LARC-Adam,learning_rate=0.0001,gradient_lag=${lag} \
                                        --scale_factor 1.0 \
                                        --batch ${batch} \
                                        --use_batchnorm \
                                        --label_id 0 \
                                        --data_format "channels_first" |& tee out.lite.fp32.lag${lag}.train.run${runid}
fi


if [ ${test} -eq 1 ]; then
  echo "Starting Evaluation"
  runid=0
  runfiles=$(ls -latr out.lite.fp32.lag${lag}.test.run* | tail -n1 | awk '{print $9}')
  if [ ! -z ${runfiles} ]; then
      runid=$(echo ${runfiles} | awk '{split($1,a,"run"); print a[1]+1}')
  fi
    
  python -u ./tiramisu-tf-inference.py     --datadir_test ${scratchdir}/validation \
                                           --test_size ${numfiles_validation} \
                                           --downsampling ${downsampling} \
                                           --downsampling_mode "center-crop" \
                                           --channels 0 1 2 10 \
                                           --chkpt_dir checkpoint.fp32.lag${lag} \
                                           --output_graph tiramisu_inference.pb \
                                           --output output_validation \
                                           --fs local \
                                           --blocks ${blocks} \
                                           --growth 32 \
                                           --filter-sz 5 \
                                           --loss weighted \
                                           --scale_factor 1.0 \
                                           --batch 5 \
                                           --use_batchnorm \
                                           --label_id 0 \
                                           --data_format "channels_first" |& tee out.lite.fp32.lag${lag}.test.run${runid}
fi


if [ ${predict} -eq 1 ]; then
  echo "Starting Prediction"
  runid=0
  runfiles=$(ls -latr out.lite.fp32.lag${lag}.test.run* | tail -n1 | awk '{print $9}')
  if [ ! -z ${runfiles} ]; then
      runid=$(echo ${runfiles} | awk '{split($1,a,"run"); print a[1]+1}')
  fi

  python -u ./tiramisu-tf-inference.py     --datadir_test ${scratchdir}/test_data \
                                           --prediction_mode \
                                           --test_size ${numfiles_test} \
                                           --downsampling ${downsampling} \
                                           --downsampling_mode "center-crop" \
										   --chkpt_dir $checkpt \
										   --output_graph tiramisu_inference.pb \
                                           --output output_test \
                                           --fs global \
										   --blocks ${blocks} \
										   --growth 32 \
										   --filter-sz 5 \
                                           --loss weighted \
                                           --scale_factor 1.0 \
                                           --batch 5 \
                                           --use_batchnorm \
                                           --label_id 0 \
                                           --data_format "channels_first" |& tee out.lite.fp32.lag${lag}.test.run${runid}
fi
