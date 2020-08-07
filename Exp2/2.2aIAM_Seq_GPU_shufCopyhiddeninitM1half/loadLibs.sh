export CUDNN_PATH=/home/videotext/src/cuda/lib64/libcudnn.so.7
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-9.0/bin/:$PATH
export LD_LIBRARY_PATH=/home/videotext/src/cuda/lib64/:$LD_LIBRARY_PATH
export PATH=/home/videotext/src/cuda/include/:$PATH
echo $PATH
echo $LD_LIBRARY_PATH
source ~/OCR/nohelmet_shubham/package/noHelmet9.1/bin/activate
