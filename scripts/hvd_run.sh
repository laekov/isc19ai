# EXPORT HOROVOD_TIMELINE=/mnt/ssd/laekov/timeline3.json 
export HOROVOD_FUSION_THRESHOLD=16777216
horovodrun -np 8 --host i2:4,i1:4 \
	$(pwd)/run_deeplab_lite.sh 

exit
