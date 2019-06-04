# EXPORT HOROVOD_TIMELINE=/mnt/ssd/laekov/timeline3.json 
export HOROVOD_FUSION_THRESHOLD=16777216
horovodrun -np 1 --host i5:1 \
	$(pwd)/run_deeplab_lite.sh 

exit
/opt/spack/opt/spack/linux-debian9-x86_64/gcc-6.3.0/openmpi-3.1.2-qubp2kdenj7feas5xliosp6av6kqizdn/bin/mpirun -np 4 --host i7:4 \
    -bind-to none -map-by slot \
	-x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
	-mca pml ob1 -mca btl ^openib \
	bash run_deeplab_lite.sh 
