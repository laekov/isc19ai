#!/bin/bash
source /etc/profile.d/modules.sh
source /opt/spack/share/spack/setup-env.sh
export PATH="/home/laekov/ssdhome/miniconda3/bin:$PATH"

#openmp stuff
export OMP_NUM_THREADS=16
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread

#pick GPU: remove for multi-gpu
# export CUDA_VISIBLE_DEVICES=0

#directories
datadir=/mnt/data
scratchdir=/mnt/ram/segm_h5_v3_new_split
# scratchdir=/mnt/ssd/ISC19_AI_DATA/segm_h5_v3_new_split
numfiles_train=3000
numfiles_validation=300
numfiles_test=500
downsampling=1
batch=2
# if [ $(hostname) = i7 ] || [ $(hostname) = i8 ]
# then
	# batch=4
# fi

#create run dir
run_dir=/mnt/ssd/laekov/deeplab_run_2
#rundir=${WORK}/data/tiramisu/runs/run_nnodes16_j6415751
mkdir -p ${run_dir}

#copy relevant files
cp ../utils/graph_flops.py ${run_dir}/
cp ../utils/common_helpers.py ${run_dir}/
cp ../utils/data_helpers.py ${run_dir}/
cp ../deeplab-tf/deeplab-tf-train.py ${run_dir}/
cp ../deeplab-tf/deeplab-tf-inference.py ${run_dir}/
cp ../deeplab-tf/deeplab_model.py ${run_dir}/

#step in
cd ${run_dir}

#some parameters
lag=1
train=1
test=0

if [ ${train} -eq 1 ]; then
  echo "Starting Training with bs = " $batch
  runid=0
  runfiles=$(ls -latr out.lite.fp32.lag${lag}.train.run* | tail -n1 | awk '{print $9}')
  if [ ! -z ${runfiles} ]; then
      runid=$(echo ${runfiles} | awk '{split($1,a,"run"); print a[1]+1}')
  fi
    
	# --channels 0 1 2 10 \

  python -u ./deeplab-tf-train.py \
    --datadir_train ${scratchdir}/train \
	--train_size ${numfiles_train} \
	--datadir_validation ${scratchdir}/validation \
	--validation_size ${numfiles_validation} \
	--downsampling ${downsampling} \
	--downsampling_mode "center-crop" \
	--chkpt_dir /mnt/ssd/laekov/checkpt2 \
	--epochs 500 \
	--fs local \
	--loss weighted_mean \
	--optimizer opt_type=LARC-Adam,learning_rate=0.0001,gradient_lag=${lag} \
	--model resnet_v2_50 \
	--scale_factor 1.0 \
	--batch ${batch} \
	--decoder bilinear \
	--device "/device:cpu:0" \
	--label_id 0 \
	--disable_imsave \
	--use_batchnorm \
	--data_format "channels_last" |& tee out.lite.fp32.lag${lag}.train.run${runid}
fi

if [ ${test} -eq 1 ]; then
  echo "Starting Testing"
  runid=0
  runfiles=$(ls -latr out.lite.fp32.lag${lag}.test.run* | tail -n1 | awk '{print $9}')
  if [ ! -z ${runfiles} ]; then
      runid=$(echo ${runfiles} | awk '{split($1,a,"run"); print a[1]+1}')
  fi
    
  python -u ./deeplab-tf-inference.py      --datadir_test ${scratchdir}/test \
                                           --test_size ${numfiles_test} \
                                           --downsampling ${downsampling} \
					   --downsampling_mode "center-crop" \
                                           --channels 0 1 2 10 \
                                           --chkpt_dir checkpoint.fp32.lag${lag} \
                                           --output_graph deepcam_inference.pb \
                                           --output output_test \
                                           --fs local \
                                           --loss weighted_mean \
                                           --model=resnet_v2_50 \
                                           --scale_factor 1.0 \
                                           --batch ${batch} \
                                           --decoder bilinear \
                                           --device "/device:cpu:0" \
                                           --label_id 0 \
					   --use_batchnorm \
                                           --data_format "channels_last" |& tee out.lite.fp32.lag${lag}.test.run${runid}
fi
