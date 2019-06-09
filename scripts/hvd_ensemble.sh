# EXPORT HOROVOD_TIMELINE=/mnt/ssd/laekov/timeline3.json 
export HOROVOD_FUSION_THRESHOLD=16777216
# horovodrun -np 1 --host i7:1 \
horovodrun -np 8 --host i7:4,i8:4 \
	$(pwd)/run_ensemble.sh 
