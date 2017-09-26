sudo apt-get -y update
sudo apt-get -y install build-essential 
#sudo apt-get -y install virtualbox-ose-guest-dkms virtualbox-ose-guest-x11 virtualbox-ose-guest-utils

sudo apt-get -y install cmake git libgtk2.0-dev pkg-config opencl-headers 
sudo apt-get -y install python-dev python-setuptools python-numpy python-matplotlib libatlas-dev libatlas3-base curl
sudo apt-get -y install python-matplotlib python-scipy python-pandas python-sympy python-nose
sudo apt-get -y install python-pip wget
sudo apt-get -y install ipython-notebook ipython 
sudo apt-get -y install ffmpeg
sudo apt-get -y install libatlas-dev libatlas3-base-dev python-matplotlib
sudo apt-get -y install pyqt4-dev-tools libicu-dev libpng12-dev libfreetype6 libfreetype6-dev libzmq-dev liblapack-dev gfortran python-qt4
sudo pip install tornado pyzmq pandas pygments matplotlib
sudo pip install pillow

sudo pip install cython
sudo pip install -U scikit-image

#scikit learn
sudo pip install -U scikit-learn

#theano
sudo pip install Theano

#genetic algorithms - deap
sudo pip install deap


#
#cd ~
#wget http://registrationcenter.intel.com/irc_nas/4181/intel_sdk_for_ocl_applications_2014_ubuntu_4.4.0.117_x64.tgz
# tar zxvf intel_sdk_for_ocl_applications_2012_x64.tgz
#sudo apt-get -y install -y rpm alien libnuma1
#fakeroot alien --to-deb intel_ocl_sdk_2012_x64.rpm
#sudo dpkg -i intel-ocl-sdk_2.0-31361_amd64.deb
#sudo ln -s /usr/lib64/libOpenCL.so /usr/lib/libOpenCL.so
#sudo ldconfig

#OpenCV
cd ~
mkdir opencv
cd opencv
git clone https://github.com/Itseez/opencv.git
cd opencv
mkdir release
cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_OPENCL=OFF -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j2
sudo make install

#Config ipython plot
mkdir ~/.matplotlib
echo "backend: Qt4Agg" >> ~/.matplotlib/matplotlibrc2

ipython profile create myserver
echo "c = get_config()" >> ~/.ipython/profile_myserver/ipython_notebook_config.py
echo "c.IPKernelApp.pylab = 'inline'"  >> ~/.ipython/profile_myserver/ipython_notebook_config.py
echo "c.NotebookApp.ip = '*'"  >> ~/.ipython/profile_myserver/ipython_notebook_config.py
echo "c.NotebookApp.open_browser = False"  >> ~/.ipython/profile_myserver/ipython_notebook_config.py
echo "c.NotebookApp.port = 8888"  >> ~/.ipython/profile_myserver/ipython_notebook_config.py



#############################################
#Torch7
#############################################
#Install Torch7 with dependencies
cd ~
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-all | bash

#torch-rocks install cwrap
#torch-rocks install paths
#torch-rocks install torch
#torch-rocks install sys
#torch-rocks install xlua
#torch-rocks install nn
#torch-rocks install nnx
#torch-rocks install optim
#torch-rocks install image
#torch-rocks install sundown
#torch-rocks install dok


#wget https://raw.github.com/torch/rocks/master/paths-scm-1.rockspec
#wget https://github.com/torch/paths/archive/master.zip
#wget https://raw.githubusercontent.com/torch/nn/master/rocks/nn-scm-1.rockspec



#export GNUTERM=x11

#cd ~/vagrant
#ipython notebook --profile=myserver

#Upgrade Ubuntu
#sudo apt-get -y upgrade
#sudo reboot 

##########################################
# DOCKER
##########################################

#sudo apt-get update
#sudo apt-get install linux-image-generic-lts-raring linux-headers-generic-lts-raring
#sudo reboot
#curl -s https://get.docker.io/ubuntu/ | sudo sh

##########################################
# UTIL
##########################################
#rsync -Cravpe ssh ./ipython guilherme@192.168.1.9:/home/guilherme/

# KILL all docker containers
#sudo docker ps -a -q | xargs sudo docker rm

#Shared Volumes
#sudo docker run -v ~/ipython:/var/host_ipython:rw -v ~/images:/var/host_images:rw opencv-2.4.8 ipython /var/host_ipython/recog_batch_ref2.py /var/host_ipython/input_ref.txt /var/host_images/caltech101

#nohup

###########################################
#SYNC
###########################################
#rsync -Cravpe ssh /Users/guilherme/code/ guilherme@10.10.10.12:/home/guilherme/code/
#rsync -Cravpe ssh /Volumes/Imagens/ guilherme@10.10.10.12:/home/guilherme/images/
#
#rsync -rav -delete --exclude=".*" ssh /Volumes/Imagens /Volumes/EX1/Images/
#rsync -rav -delete ssh /Volumes/Dados /Volumes/EX1/Dados/

##########################################
# OpenBLAS
##########################################
# cd ~/code/bag_of_features/overfeat/OpenBLAS
# make
# sudo make install
###########################################
# LOCALES
# sudo dpkg-reconfigure locales
###########################################
# ACESSO
# ngrok
# wget https://dl.ngrok.com/linux_386/ngrok.zip

##########################################
# Overfeat
##########################################
# cd ~/code/bag_of_features/overfeat/overfeat/src
# make clean
# make all
# cd ~/code/bag_of_features/overfeat/overfeat/API/python
# sudo python setup.py install
# 
#####################################################
#
#####################################################
# n_img = 500
# n_desc = 50000
# img_size = [64, 128, 256, 512]
# tamanhos k = [100, 500, 1000, 1500]
# classificadores = [Kmeans+SVM, Kmeans+OPF, OPF+SVM, OPF+OPF, Overfeat+SVM, Overfeat+OPF, cnn]
# bases = [caltech101, caltech256, cifar10, gui001]
# tamanhos gui001 = [35, 30, 25, 20, 15, 10, 5]
# descritores = [SIFT+SIFT, MSER+SIFT, Dense+SIFT, SURF+SURF, Dense+SURF]
# analise = [acuracia, tempo geral, tempo classificacao, tempo clusterizacao]
#
####################################################
# PLOTS 
# K vs Acuracia
# K vs F1
# BOVW (MSER+SIFT), CNN, OVERFEAT vs Acuracia (3 bases)
# BOVW (MSER+SIFT), CNN, OVERFEAT vs F1 (3 bases)
# Tempo agrupamento 
# Tempo total
#
#
#python recog.py -i input.txt -dir_train /home/guilherme/images/CALTECH101/train -dir_test /home/guilherme/images/CALTECH101/test -dir_results /home/guilherme/results/result_caltech101/
#
# DIA 1 - Segunda
# recog02 - overfeat - caltech101
# recog03 - overfeat - caltech256
# recog04 - overfeat - CIFAR10
# recog05 - bow 100 - CALTECH101
# recog06 - desativada
#
# DIA 2 - Terca
# recog02 - bow 200 - CALTECH101
# recog03 - overfeat - caltech256
# recog04 - overfeat - CIFAR10
# recog05 - bow 200 - CALTECH101
# recog06 - bow 300 - CALTECH101
#
# DIA 3 - Quarta
# recog02 - bow 200 - CALTECH101
# recog03 - cnn - model 1 - CALTECH101_52
# recog04 - overfeat - CIFAR10
# recog05 - bow 200 - CALTECH101
# recog06 - bow 300 - CALTECH101 (erro)
#
# DIA 4 - Quinta
# recog02 - bow 100 - CALTECH101
# recog03 - cnn - model 1 - CALTECH101_52
# recog04 - overfeat - CIFAR10
# recog05 - bow 250 - CALTECH101
# recog06 - cnn - model 4 - CALTECH101_52
#
# DIA 5 - Sexta
# recog02 - nohup python recog.py -i input_opf_opf.txt -dir_train /home/guilherme/images/CALTECH101/train -dir_test /home/guilherme/images/CALTECH101/test -dir_results /home/guilherme/results/result_opf_opf_caltech101/ &
# recog03 - cnn - model 4 - CALTECH256_52
# recog04 - nohup python recog.py -i input_kmeans_opf.txt -dir_train /home/guilherme/images/CALTECH101/train -dir_test /home/guilherme/images/CALTECH101/test -dir_results /home/guilherme/results/result_kmeans_opf_caltech101/ &
# recog05 - nohup python recog.py -i input_kmeans_svm.txt -dir_train ~/images/CALTECH101/train -dir_test ~/images/CALTECH101/test -dir_results ~/results/result_kmeans_svm_caltech101/ &
# recog06 - cnn - model 4 - CALTECH101_52
#
###################################################################
# SEMANA 2
###################################################################
# DIA 2 - Terca feira
# recog02 - nohup python recog.py -i input_kmeans_svm.txt -dir_train ~/images/CALTECH256/train -dir_test ~/images/CALTECH256/test -dir_results ~/results/caltech256/ &
# recog03 - nohup python recog.py -i input_kmeans_opf.txt -dir_train ~/images/CALTECH256/train -dir_test ~/images/CALTECH256/test -dir_results ~/results/caltech256/ &
# recog04 - nohup python recog.py -i input_opf_opf.txt -dir_train ~/images/CALTECH256/train -dir_test ~/images/CALTECH256/test -dir_results ~/results/caltech256/ &
# recog05 - nohup python recog.py -i input_opf_svm.txt -dir_train ~/images/CALTECH256/train -dir_test ~/images/CALTECH256/test -dir_results ~/results/caltech256/ &
# recog06 - nohup python recog.py -i input_desc_size.txt -dir_train ~/images/CALTECH256/train -dir_test ~/images/CALTECH256/test -dir_results ~/results/caltech256/ &
#
# DIA 3 - Quarta Feira - Igual terca
#
# DIA 4 - Quinta feira
# recog02 - nohup python recog.py -i input_kmeans_svm.txt -dir_train ~/images/CALTECH101/train -dir_test ~/images/CALTECH101/test -dir_results ~/results/caltech101/ &
# recog03 - nohup python recog.py -i input_kmeans_opf.txt -dir_train ~/images/CALTECH101/train -dir_test ~/images/CALTECH101/test -dir_results ~/results/CALTECH101/ &
# recog04 - nohup python recog.py -i input_opf_opf.txt -dir_train ~/images/CALTECH101/train -dir_test ~/images/CALTECH101/test -dir_results ~/results/CALTECH101/ &
# recog05 - nohup python recog.py -i input_opf_svm.txt -dir_train ~/images/CALTECH101/train -dir_test ~/images/CALTECH101/test -dir_results ~/results/CALTECH101/ &
# recog06 - nohup python recog.py -i input_desc_size.txt -dir_train ~/images/CALTECH256/train -dir_test ~/images/CALTECH256/test -dir_results ~/results/caltech256/ &
#
# DIA 5 - Sexta feira
# recog02 - nohup python recog.py -i input_kmeans_svm.txt -dir_train ~/images/GUI001/train -dir_test ~/images/GUI001/test -dir_results ~/results/gui001/ &
# recog03 - nohup python recog.py -i input_kmeans_opf.txt -dir_train ~/images/GUI001/train -dir_test ~/images/GUI001/test -dir_results ~/results/gui001/ &
# recog04 - nohup python recog.py -i input_opf_opf.txt -dir_train ~/images/CALTECH101/train -dir_test ~/images/CALTECH101/test -dir_results ~/results/CALTECH101/ &
# recog05 - nohup python recog.py -i input_opf_svm.txt -dir_train ~/images/CALTECH101/train -dir_test ~/images/CALTECH101/test -dir_results ~/results/CALTECH101/ &
# recog06 - nohup python recog.py -i input_desc_size.txt -dir_train ~/images/CALTECH256/train -dir_test ~/images/CALTECH256/test -dir_results ~/results/caltech256/ &
#
# DIA 6 - Segunda Feira
# recog02 - nohup python recog.py -i input_opf_opf.txt -dir_train ~/images/GUI001/train -dir_test ~/images/GUI001/test -dir_results ~/results/gui001_2/ &
# recog03 - nohup python recog.py -i input_opf_svm.txt -dir_train ~/images/GUI001/train -dir_test ~/images/GUI001/test -dir_results ~/results/gui001_2/ &
# recog04 - nohup python recog.py -i input_opf_opf.txt -dir_train ~/images/CALTECH101/train -dir_test ~/images/CALTECH101/test -dir_results ~/results/CALTECH101/ &
# recog05 - nohup python recog.py -i input_opf_svm.txt -dir_train ~/images/CALTECH101/train -dir_test ~/images/CALTECH101/test -dir_results ~/results/CALTECH101/ &
# recog06 - nohup python recog.py -i input_desc_size.txt -dir_train ~/images/CALTECH256/train -dir_test ~/images/CALTECH256/test -dir_results ~/results/caltech256/ &
# 
#
# DIA 7 - Terca feira
# recog02 - nohup python recog.py -i input_opf_opf.txt -dir_train ~/images/GUI001/train -dir_test ~/images/GUI001/test -dir_results ~/results/gui001_2/ &
# recog03 - nohup python recog.py -i input_opf_svm.txt -dir_train ~/images/GUI001/train -dir_test ~/images/GUI001/test -dir_results ~/results/gui001_2/ &
# recog04 - nohup python recog.py -i input_opf_opf.txt -dir_train ~/images/CALTECH101/train -dir_test ~/images/CALTECH101/test -dir_results ~/results/CALTECH101/ &
# recog05 - nohup python recog.py -i input_opf_svm.txt -dir_train ~/images/CALTECH101/train -dir_test ~/images/CALTECH101/test -dir_results ~/results/CALTECH101/ &
# #trocado recog06 - nohup python recog.py -i input_desc_size.txt -dir_train ~/images/CALTECH101/train -dir_test ~/images/CALTECH101/test -dir_results ~/results/caltech101/ &
# 
# Coletar resultados da recog02 e recog03 gui001
# DIA 1 - Segunda feira
# recog02 - nohup python recog.py -i input_opf_opf.txt -dir_train ~/images/GUI001/train -dir_test ~/images/GUI001/test -dir_results ~/results/gui001_2/ &
# recog03 - nohup python recog.py -i input_opf_svm.txt -dir_train ~/images/GUI001/train -dir_test ~/images/GUI001/test -dir_results ~/results/gui001_2/ &
# recog04 - nohup python recog.py -i input_opf_opf.txt -dir_train ~/images/CALTECH101/train -dir_test ~/images/CALTECH101/test -dir_results ~/results/CALTECH101/ &
# recog05 - python recog.py -i input_k_size.txt -dir_train /home/guilherme/images/CALTECH101/train -dir_test /home/guilherme/images/CALTECH101/test -dir_results /home/guilherme/results/CALTECH101_k_size/
# recog06 - cnn cifar10
#
# DIA 2 - Terca feira
# recog02 - nohup python recog.py -i input_opf_opf.txt -dir_train ~/images/GUI001/train -dir_test ~/images/GUI001/test -dir_results ~/results/gui001_2/ &
# recog03 - nohup python recog.py -i input_opf_svm.txt -dir_train ~/images/GUI001/train -dir_test ~/images/GUI001/test -dir_results ~/results/gui001_2/ &
# recog04 - overfeat caltech101
# recog05 - python recog.py -i input_k_size.txt -dir_train /home/guilherme/images/CALTECH101/train -dir_test /home/guilherme/images/CALTECH101/test -dir_results /home/guilherme/results/CALTECH101_k_size/
# recog06 - overfeat gui001
#
# DIA 2 - Quarta feira
# recog02 - nohup python recog.py -i input_opf_opf.txt -dir_train ~/images/GUI001/train -dir_test ~/images/GUI001/test -dir_results ~/results/gui001_2/ &
# recog03 - nohup python recog.py -i input_opf_svm.txt -dir_train ~/images/GUI001/train -dir_test ~/images/GUI001/test -dir_results ~/results/gui001_2/ &
# recog04 - overfeat caltech101
# recog05 - python recog.py -i input_k_size.txt -dir_train /home/guilherme/images/GUI001/train -dir_test /home/guilherme/images/GUI001/test -dir_results /home/guilherme/results/gui001_k_size/
# recog06 - overfeat gui001
#
# DIA 2 - Quinta feira
# recog02 - nohup python recog.py -i input_opf_opf.txt -dir_train ~/images/GUI001/train -dir_test ~/images/GUI001/test -dir_results ~/results/gui001_2/ &
# recog03 - nohup python recog.py -i input_opf_svm.txt -dir_train ~/images/GUI001/train -dir_test ~/images/GUI001/test -dir_results ~/results/gui001_2/ &
# recog04 - overfeat cifar10
# recog05 - python recog.py -i input_k_size.txt -dir_train /home/guilherme/images/GUI001/train -dir_test /home/guilherme/images/GUI001/test -dir_results /home/guilherme/results/gui001_k_size/
# recog06 - 
#
# DIA 2 - Sexta feira
# recog02 - nohup python recog.py -i input_opf_opf_param.txt -dir_train /home/guilherme/images/GUI001/train -dir_test /home/guilherme/images/GUI001/test -dir_results ~/results/gui001_opf_param/ &
# recog03 - nohup python recog.py -i input_opf_svm_param.txt -dir_train /home/guilherme/images/GUI001/train -dir_test /home/guilherme/images/GUI001/test -dir_results ~/results/gui001_opf_param/ &
# recog04 - overfeat cifar10
# recog05 - nohup python recog.py -i input_opf_svm_param.txt -dir_train /home/guilherme/images/CALTECH101/train -dir_test /home/guilherme/images/CALTECH101/test -dir_results ~/results/caltech101_opf_param/ &
# recog06 - nohup python recog.py -i input_opf_opf_param.txt -dir_train /home/guilherme/images/CALTECH101/train -dir_test /home/guilherme/images/CALTECH101/test -dir_results /home/guilherme/results/caltech101_opf_param/ &
#
# ###############################################################
# DIA 1 - Segunda feira
# recog02 - python recog.py -i input_opf_opf_param.txt -dir_train /home/guilherme/images/GUI001/train/ -dir_test /home/guilherme/images/GUI001/test -dir_results /home/guilherme/results/gui001_opf_opf_param/
# recog03 - python recog.py -i input_opf_svm_param.txt -dir_train /home/guilherme/images/GUI001/train/ -dir_test /home/guilherme/images/GUI001/test -dir_results /home/guilherme/results/gui001_opf_svm_param/
# recog04 - nohup python recog.py -i input_opf_opf_param.txt -dir_train ~/images/CALTECH101/train -dir_test ~/images/CALTECH101/test -dir_results ~/results/caltech101_opf_opf_params/ &
# recog05 - nohup python recog.py -i input_opf_svm_param.txt -dir_train ~/images/CALTECH101/train -dir_test ~/images/CALTECH101/test -dir_results ~/results/caltech101_opf_svm_params/ &
# recog06 - 
#
#####################################################################
# 16/06 - Segunda feira
# recog02 - python recog.py -i input_gui001_kmeans_svm.txt -dir_train /home/guilherme/images/GUI001/train/ -dir_test /home/guilherme/images/GUI001/test -dir_results /home/guilherme/results/results_gui001_kmeans_svm/
# recog03 - python recog.py -i input_gui001_kmeans_opf.txt -dir_train /home/guilherme/images/GUI001/train/ -dir_test /home/guilherme/images/GUI001/test/ -dir_results /home/guilherme/results/results_gui001_kmeans_opf/
# recog04 - python recog.py -i input_caltech101_kmeans_svm.txt -dir_train /home/guilherme/images/CALTECH101/train/ -dir_test /home/guilherme/images/CALTECH101/test/ -dir_results /home/guilherme/results/results_caltech101_kmeans_svm/
# recog05 - python recog.py -i input_caltech101_kmeans_opf.txt -dir_train /home/guilherme/images/CALTECH101/train/ -dir_test /home/guilherme/images/CALTECH101/test/ -dir_results /home/guilherme/results/results_caltech101_kmeans_opf/
# recog06 - python recog.py -i input_gui001_opf_svm.txt -dir_train ~/images/GUI001/train/ -dir_test ~/images/GUI001/test/ -dir_results /home/guilherme/results/results_gui001_opf_svm/
#
# 17/06 - Terca feira
# recog02 - input_opf_opf_param - GUI001_5
# recog03 - input_opf_svm_param - GUI001_5
# recog04 - input_k_size_kmeans - GUI001
# recog05 - input_k_size_kmeans - CALTECH101
# recog06 - input_opf_opf_param - GUI001_10
#
# 18/06 - Quarta feira
# recog02 - python recog.py -i input_caltech101_opf_svm.txt -dir_train /home/guilherme/images/CALTECH101/train/ -dir_test /home/guilherme/images/CALTECH101/test/ -dir_results /home/guilherme/results/results_caltech101_opf_svm/
# recog03 - python recog.py -i input_opf_svm_param.txt -dir_train /home/guilherme/images/GUI001_5/train/ -dir_test /home/guilherme/images/GUI001_5/test/ -dir_results /home/guilherme/results/results_gui001_5_opf_svm_param/
# recog04 - python recog.py -i input_gui001_opf_opf.txt -dir_train /home/guilherme/images/GUI001/train/ -dir_test /home/guilherme/images/GUI001/test/ -dir_results /home/guilherme/results/results_gui001_opf_opf/
# recog05 - python recog.py -i input_gui001_opf_svm.txt -dir_train /home/guilherme/images/GUI001/train/ -dir_test /home/guilherme/images/GUI001/test/ -dir_results /home/guilherme/results/results_gui001_opf_svm/
# recog06 - python recog.py -i input_caltech101_opf_opf.txt -dir_train /home/guilherme/images/CALTECH101/train/ -dir_test /home/guilherme/images/CALTECH101/test/ -dir_results 
#
# 23/06 - Segunda Feira
# COL - recog02 - python recog.py -i input_overfeat.txt -dir_train /home/guilherme/images/GUI001_5/train -dir_test /home/guilherme/images/GUI001_5/test -dir_results /home/guilherme/results/results_gui001_5_overfeat/
# recog02 - python recog.py -i input_overfeat.txt -dir_train /home/guilherme/images/GUI001_10/train -dir_test /home/guilherme/images/GUI001_10/test -dir_results /home/guilherme/results/results_gui001_10_overfeat/ &
# COL - recog03 - python recog.py -i input_gui001_5_opf_opf.txt -dir_train /home/guilherme/images/GUI001_5/train/ -dir_test /home/guilherme/images/GUI001_5/test/ -dir_results /home/guilherme/results/results_gui001_5_opf_opf/
# recog03 - python recog.py -i input_opf_opf_param.txt -dir_train /home/guilherme/images/GUI001_15/train/ -dir_test /home/guilherme/images/GUI001_15/test/ -dir_results /home/guilherme/results/results_gui001_15_opf_opf_param/ &
# COL - recog04 - python recog.py -i input_gui001_opf_svm.txt -dir_train /home/guilherme/images/GUI001_5/train/ -dir_test /home/guilherme/images/GUI001_5/test -dir_results /home/guilherme/results/results_gui001_5_opf_svm/
# recog04 - python recog.py -i input_opf_svm_param.txt -dir_train /home/guilherme/images/GUI001_15/train/ -dir_test /home/guilherme/images/GUI001_15/test/ -dir_results /home/guilherme/results/results_gui001_15_opf_svm_param/ &
# COL - recog05 - python recog.py -i input_k_size_kmeans.txt -dir_train /home/guilherme/images/GUI001_5/ -dir_test /home/guilherme/images/GUI001_5/test -dir_results /home/guilherme/results/results_gui001_5_k_size_kmeans/
# recog05 - python recog.py -i input_opf_opf_param.txt -dir_train /home/guilherme/images/GUI001_10/train/ -dir_test /home/guilherme/images/GUI001_10/test/ -dir_results /home/guilherme/results/results_gui001_10_opf_opf_param
# recog06 - python recog.py -i input_gui001_5_kmeans_opf.txt -dir_train /home/guilherme/images/GUI001_5/train/ -dir_test /home/guilherme/images/GUI001_5/test/ -dir_results /home/guilherme/results/results_gui001_5_kmeans_opf/
# COL - recog06 - python recog.py -i input_gui001_5_kmeans_svm.txt -dir_train /home/guilherme/images/GUI001_5/train/ -dir_test /home/guilherme/images/GUI001_5/test/ -dir_results /home/guilherme/results/results_gui001_5_kmeans_opf/
# recog06 - python recog.py -i input_opf_svm_param.txt -dir_train /home/guilherme/images/GUI001_10/train/ -dir_test /home/guilherme/images/GUI001_10/test/ -dir_results /home/guilherme/results/results_gui001_10_opf_svm_param/ 
#
# 24/06 - Terça Feira
# recog02 - python recog.py -i input_overfeat.txt -dir_train /home/guilherme/images/GUI001_15/train -dir_test /home/guilherme/images/GUI001_15/test -dir_results /home/guilherme/results/results_gui001_15_overfeat/
# recog03 - python recog.py -i input_opf_opf_param.txt -dir_train /home/guilherme/images/GUI001_15/train/ -dir_test /home/guilherme/images/GUI001_15/test/ -dir_results /home/guilherme/results/results_gui001_15_opf_opf_param/
# recog04 - python recog.py -i input_opf_svm_param.txt -dir_train /home/guilherme/images/GUI001_15/train/ -dir_test /home/guilherme/images/GUI001_15/test/ -dir_results /home/guilherme/results/results_gui001_15_opf_svm_param/
# recog05 - python recog.py -i input_opf_opf_param.txt -dir_train /home/guilherme/images/GUI001_10/train/ -dir_test /home/guilherme/images/GUI001_10/test/ -dir_results /home/guilherme/results/results_gui001_10_opf_opf_param
# recog06 - python recog.py -i input_opf_svm_param.txt -dir_train /home/guilherme/images/GUI001_10/train/ -dir_test /home/guilherme/images/GUI001_10/test/ -dir_results /home/guilherme/results/results_gui001_10_opf_svm_param/
#
# 25/06 - Quarta Feira
# COL - recog02 - python recog.py -i input_overfeat.txt -dir_train /home/guilherme/images/GUI001_15/train -dir_test /home/guilherme/images/GUI001_15/test -dir_results /home/guilherme/results/results_gui001_15_overfeat/
# recog02 - python recog.py -i input_opf_opf_param.txt -dir_train /home/guilherme/images/CIFAR10/train/ -dir_test /home/guilherme/images/CIFAR10/test/ -dir_results /home/guilherme/results/results_cifar10_opf_opf_param/ 
# recog03 - python recog.py -i input_opf_opf_param.txt -dir_train /home/guilherme/images/GUI001_15/train/ -dir_test /home/guilherme/images/GUI001_15/test/ -dir_results /home/guilherme/results/results_gui001_15_opf_opf_param/
# recog04 - python recog.py -i input_opf_svm_param.txt -dir_train /home/guilherme/images/GUI001_15/train/ -dir_test /home/guilherme/images/GUI001_15/test/ -dir_results /home/guilherme/results/results_gui001_15_opf_svm_param/
# COL - recog05 - python recog.py -i input_opf_opf_param.txt -dir_train /home/guilherme/images/GUI001_10/train/ -dir_test /home/guilherme/images/GUI001_10/test/ -dir_results /home/guilherme/results/results_gui001_10_opf_opf_param
# COL - recog05 - nohup python recog.py -i input_k_size_kmeans.txt -dir_train ~/images/GUI001_15/train/ -dir_test ~/images/GUI001_15/test/ -dir_results ~/results/results_gui001_15_k_size_kmeans/ &
# recog05 - nohup python recog.py -i input_gui001_10_kmeans_opf.txt -dir_train /home/guilherme/images/GUI001_10/train/ -dir_test /home/guilherme/images/GUI001_10/test/ -dir_results /home/guilherme/results/results_gui001_10_kmeans_opf/ &
# COL - recog06 - python recog.py -i input_opf_svm_param.txt -dir_train /home/guilherme/images/GUI001_10/train/ -dir_test /home/guilherme/images/GUI001_10/test/ -dir_results /home/guilherme/results/results_gui001_10_opf_svm_param/
# COL - recog06 - nohup python recog.py -i input_k_size_kmeans.txt -dir_train ~/images/GUI001_10/train/ -dir_test ~/images/GUI001_10/test/ -dir_results ~/results/results_gui001_10_k_size_kmeans/ &
# recog06 - nohup python recog.py -i input_gui001_10_kmeans_svm.txt -dir_train /home/guilherme/images/GUI001_10/train/ -dir_test /home/guilherme/images/GUI001_10/test/ -dir_results /home/guilherme/results/results_gui001_10_kmeans_svm/ &
# 
# 26/06 - Quinta Feira
# recog02 - python recog.py -i input_opf_opf_param.txt -dir_train /home/guilherme/images/CIFAR10/train/ -dir_test /home/guilherme/images/CIFAR10/test/ -dir_results /home/guilherme/results/results_cifar10_opf_opf_param/ 
# recog03 - python recog.py -i input_gui001_15_kmeans_opf.txt -dir_train /home/guilherme/images/GUI001_15/train/ -dir_test /home/guilherme/images/GUI001_15/test/ -dir_results /home/guilherme/results/results_gui001_15_kmeans_opf/
# recog04 - python recog.py -i input_gui001_15_kmeans_svm.txt -dir_train /home/guilherme/images/GUI001_15/train/ -dir_test /home/guilherme/images/GUI001_15/test/ -dir_results /home/guilherme/results/results_gui001_15_kmeans_svm/
# recog05 - python recog.py -i input_gui001_10_opf_opf.txt -dir_train /home/guilherme/images/GUI001_10/train/ -dir_test /home/guilherme/images/GUI001_10/test/ -dir_results /home/guilherme/results/results_gui001_10_opf_opf/
# recog06 - python recog.py -i input_gui001_10_opf_svm.txt -dir_train /home/guilherme/images/GUI001_10/train/ -dir_test /home/guilherme/images/GUI001_10/test/ -dir_results /home/guilherme/results/results_gui001_10_opf_svm/ 
#
# 27/06 - Sexta feira
# recog02 - python recog.py -i input_opf_opf_param.txt -dir_train /home/guilherme/images/CIFAR10/train/ -dir_test /home/guilherme/images/CIFAR10/test/ -dir_results /home/guilherme/results/results_cifar10_opf_opf_param/ 
# recog03 - 
# recog04 - python recog.py -i input_gui001_15_opf_opf.txt -dir_train ~/images/GUI001_15/train/ -dir_test ~/images/GUI001_15/test/ -dir_results ~/results/results_gui001_15_opf_opf/
# recog05 - python recog.py -i input_gui001_15_opf_svm.txt -dir_train ~/images/GUI001_15/train/ -dir_test ~/images/GUI001_15/test/ -dir_results ~/results/results_gui001_15_opf_svm/ 
# recog06 -
#
# 03/07 - Quinta Feira
# OK - recog02 - cnn rgb gui001
# OK - COL - recog03 - cnn yuv gui001_15
# OK - COL - recog04 - cnn rgb gui001_5 | cnn rgb gui001_10
# COL - recog05 - cnn rgb gui001_15
# recog06 - cnn rgb caltech101 