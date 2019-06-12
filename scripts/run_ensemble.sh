#!/bin/bash
source /etc/profile.d/modules.sh
source /opt/spack/share/spack/setup-env.sh
export PATH="/home/laekov/ssdhome/miniconda3/bin:$PATH"

#openmp stuff
export OMP_NUM_THREADS=16
# export OMP_PLACES=sockets
# export OMP_PROC_BIND=spread

#pick GPU: remove for multi-gpu
# export CUDA_VISIBLE_DEVICES=0

#directories
datadir=/mnt/data
# scratchdir=/mnt/ram/segm_h5_v3_new_split
scratchdir=/mnt/data/segm_h5_v3_new_split
# scratchdir=/mnt/ssd/ISC19_AI_DATA/segm_h5_v3_new_split
checkpt=/mnt/ssd/laekov/checkpt_ds
numfiles_train=3000
numfiles_validation=300
numfiles_test=500
downsampling=4
batch=1
if [ $(hostname) = i7 ] || [ $(hostname) = i8 ]
then
	batch=62
fi

export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
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


#create run dir
run_dir=/mnt/ssd/laekov/deeplab_run
#run_dir=${WORK}/data/tiramisu/runs/run_nnodes16_j6415751
mkdir -p ${run_dir}

#copy relevant files
cp ../utils/graph_flops.py ${run_dir}/
cp ../utils/common_helpers.py ${run_dir}/
cp ../utils/data_helpers.py ${run_dir}/
cp ../deeplab-tf/deeplab-tf-train.py ${run_dir}/
cp ../deeplab-tf/ensemble-tf-inference.py ${run_dir}/
cp ../deeplab-tf/deeplab_model.py ${run_dir}/
cp ../tiramisu-tf/tiramisu_model.py ${run_dir}/

#step in
cd ${run_dir}

#some parameters
lag=1
train=0
test=1

if [ ${test} -eq 1 ]; then
  echo "Starting Testing with batch size = " $batch
  runid=0
  runfiles=$(ls -latr out.lite.fp32.lag${lag}.test.run* | tail -n1 | awk '{print $9}')
  if [ ! -z ${runfiles} ]; then
      runid=$(echo ${runfiles} | awk '{split($1,a,"run"); print a[1]+1}')
  fi
    
  # numactl --cpunodebind=$socket -m $socket \
  python -u ./ensemble-tf-inference.py \
		--datadir_test ${scratchdir}/test \
		--test_size ${numfiles_test} \
		--downsampling ${downsampling} \
		--downsampling_mode "center-crop" \
		--chkpt_dir $checkpt \
		--output_graph deepcam_inference.pb \
		--output output_test \
		--fs global \
		--loss weighted_mean \
		--model=resnet_v2_101 \
		--scale_factor 1.0 \
		--batch ${batch} \
		--decoder bilinear \
		--device "/device:cpu:0" \
		--label_id 0 \
		--use_batchnorm \
		--data_format "channels_last" |& tee out.lite.fp32.lag${lag}.test.run${runid}
fi
