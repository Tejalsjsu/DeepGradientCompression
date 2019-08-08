startTime=${SECONDS}

sudo rm -r mnist_convnet_model
source activate tensorflow_p36


mpi_command="/home/ubuntu/anaconda3/envs/tensorflow_p36/bin/mpirun -np 16 -H localhost:8  \
--allow-run-as-root -bind-to none -map-by slot \
-x NCCL_DEBUG=INFO \
-x LD_LIBRARY_PATH \
-x NCCL_SOCKET_IFNAME=ens5 -mca btl_tcp_if_exclude lo,docker0 \
-x PATH -mca pml ob1 -mca btl ^openib"
use_hvd="--horovod"


$mpi_command python mnist_estimator.py


endTime=${SECONDS}
diffTime=`expr ${endTime} - ${startTime}`
echo "Diff Time: [${diffTime}]":
