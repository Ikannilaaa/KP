sudo apt-get update -y && sudo apt-get install pip -y

apt-get -y --no-install-recommends upgrade && \
	apt-get install -y --no-install-recommends \
	nvidia-cuda-toolkit \
	build-essential \
	cmake \
	git \
	libatlas-base-dev \
	libprotobuf-dev \
	libleveldb-dev \
	libsnappy-dev \
	libhdf5-serial-dev \
	protobuf-compiler \
	libboost-all-dev \
	libgflags-dev \
	libgoogle-glog-dev \
	liblmdb-dev \
	pciutils \
	python3-setuptools \
	python3-dev \
	python3-pip \
	python3 \
	python3-distutils \
	opencl-headers \
	ocl-icd-opencl-dev \
	libviennacl-dev \
	libcanberra-gtk-module \
	libopencv-dev \
	unzip \
	nano

python3 -m pip install \
	numpy \
	protobuf \
	opencv-python \
	gdown

gdown https://drive.google.com/uc?id=1O2mW69xf6wYw6ahb1lTkJROfXeR9v9In

tar -xvf cudnn-linux-x86_64-8.3.3.40_cuda11.5-archive.tar.xz
cd cudnn-linux-x86_64-8.3.3.40_cuda11.5-archive

sudo cp include/cudnn*.h /usr/include
sudo cp lib/libcudnn* /usr/lib/x86_64-linux-gnu
sudo chmod a+r /usr/include/cudnn*.h /usr/lib/x86_64-linux-gnu/libcudnn*
sudo ldconfig

cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

cd ~

git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

wget "https://drive.usercontent.google.com/download?id=1QCSxJZpnWvM00hx49CJ2zky7PWGzpcEh&export=download&authuser=0&confirm=t&uuid=50355e54-0445-4875-9d0e-3866a1f6239d&at=APZUnTVA3sHsK5rk-u2O6phIms2L%3A1711987634035" -O models.zip

unzip models.zip -d openpose

# Comment out getModels.sh executions (Jetson TX2 related scripts)
sed -i 's/executeShInItsFolder "getModels.sh"/# executeShInItsFolder "getModels.sh"/g' ./openpose/scripts/ubuntu/install_openpose_JetsonTX2_JetPack3.1.sh
sed -i 's/executeShInItsFolder "getModels.sh"/# executeShInItsFolder "getModels.sh"/g' ./openpose/scripts/ubuntu/install_openpose_JetsonTX2_JetPack3.3.sh

# Comment out download_model and checksum lines in CMakeLists.txt
sed -i 's/download_model("BODY_25"/# download_model("BODY_25"/g' ./openpose/CMakeLists.txt
sed -i 's/78287B57CF85FA89C03F1393D368E5B7/# 78287B57CF85FA89C03F1393D368E5B7/g' ./openpose/CMakeLists.txt
sed -i 's/download_model("body (COCO)"/# download_model("body (COCO)"/g' ./openpose/CMakeLists.txt
sed -i 's/5156d31f670511fce9b4e28b403f2939/# 5156d31f670511fce9b4e28b403f2939/g' ./openpose/CMakeLists.txt
sed -i 's/download_model("body (MPI)"/# download_model("body (MPI)"/g' ./openpose/CMakeLists.txt
sed -i 's/2ca0990c7562bd7ae03f3f54afa96e00/# 2ca0990c7562bd7ae03f3f54afa96e00/g' ./openpose/CMakeLists.txt
sed -i 's/download_model("face"/# download_model("face"/g' ./openpose/CMakeLists.txt
sed -i 's/e747180d728fa4e4418c465828384333/# e747180d728fa4e4418c465828384333/g' ./openpose/CMakeLists.txt
sed -i 's/download_model("hand"/# download_model("hand"/g' ./openpose/CMakeLists.txt
sed -i 's/a82cfc3fea7c62f159e11bd3674c1531/# a82cfc3fea7c62f159e11bd3674c1531/g' ./openpose/CMakeLists.txt

mkdir -p openpose/build && \
	cd openpose/build && \
	cmake .. -DUSE_CUDNN=OFF -DGENERATE_PYTHON_BINDINGS:BOOL="1" -DBUILD_PYTHON=ON && \
	make -j`nproc`

sudo make install
export PYTHONPATH=/usr/local/python:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
echo 'export PYTHONPATH=/usr/local/python:$PYTHONPATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

cd ~

# Untuk install backend
pip3 install flask cloudpickle pandas scikit-learn joblib flask-socketio eventlet flask-cors

mkdir skripsi && cd skripsi

git clone https://github.com/Danar1111/Openpose-Backend.git

cd Openpose-Backend/

curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

sudo npm install -g pm2

pm2 start run.py --name openpose-backend --interpreter /usr/bin/python3.8

# python3 run.py