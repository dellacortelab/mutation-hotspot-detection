FROM nvidia/cuda:9.2-devel-centos7

RUN yum makecache -y && \
     yum install -y epel-release wget cmake vim octave && \
     yum makecache -y && \
     yum install -y conda sudo python36-pip

RUN pip3 install --upgrade pip && pip3 install \
        ipython \
        matplotlib \
        numpy \
        pyfiglet \
        requests==2.22.0 \
        scipy \ 
	    sentencepiece \
        torch \
        transformers