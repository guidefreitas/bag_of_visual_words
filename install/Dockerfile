#docker build -t guidefreitas/ipython_opencv-2.4.8 -< Dockerfile

FROM saltstack/ubuntu-12.04-minimal
RUN apt-get -y update
RUN apt-get -y install build-essential 
RUN apt-get -y install cmake git libgtk2.0-dev pkg-config
RUN apt-get -y install python-dev python-setuptools python-numpy python-matplotlib 
RUN apt-get -y install python-matplotlib python-scipy python-pandas python-sympy python-nose
RUN apt-get -y install python-pip
RUN apt-get -y install ipython-notebook ipython 
RUN apt-get -y install ffmpeg
RUN apt-get -y install libatlas-dev libatlas3gf-base python-matplotlib
RUN apt-get -y install pyqt4-dev-tools libicu-dev 
RUN apt-get -y install libpng12-dev libfreetype6 libfreetype6-dev 
RUN apt-get -y install libzmq-dev liblapack-dev gfortran python-qt4
RUN pip install tornado pyzmq pandas pygments matplotlib
RUN pip install pillow
RUN pip install -U scikit-learn
RUN pip install Theano
RUN pip install deap

RUN git clone https://github.com/Itseez/opencv.git
RUN (cd opencv; mkdir release; cd release; cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_OPENCL=OFF -D CMAKE_INSTALL_PREFIX=/usr/local ..; make -j2; sudo make install)
RUN rm -rf opencv
