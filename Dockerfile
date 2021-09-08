FROM dhealth/pylibs-toolkit:0.10.1-cudnn

# install cassandra C++ driver
RUN \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get update -y -q \
    && apt-get install -y libuv1-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/* 

RUN \
    wget 'https://github.com/datastax/cpp-driver/archive/2.16.0.tar.gz' \
    && tar xfz 2.16.0.tar.gz \
    && cd cpp-driver-2.16.0 \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j \
    && make install


#install cassandra python driver
RUN \
    pip3 install --upgrade --no-cache pillow \
    && pip3 install --upgrade --no-cache tqdm numpy matplotlib \
    && pip3 install --upgrade --no-cache opencv-python matplotlib \
    && pip3 install --upgrade --no-cache cassandra-driver 

# install some useful tools
RUN \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get update -y -q \
    && apt-get install -y \
       aptitude \
       bash-completion \
       dnsutils \
       elinks \
       emacs25-nox emacs-goodies-el \
       fish \
       git \
       htop \
       iproute2 \
       iputils-ping \
       ipython3 \
       less \
       mc \
       nload \
       nmon \
       psutils \
       source-highlight \
       tmux \
       vim \
       wget \
    && rm -rf /var/lib/apt/lists/*

COPY . /cassandradl
WORKDIR /cassandradl
RUN pip3 install .
