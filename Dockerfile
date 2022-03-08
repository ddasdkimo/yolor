FROM nvcr.io/nvidia/pytorch:21.09-py3

# apt install required packages
RUN apt update
RUN apt install -y zip htop screen libgl1-mesa-glx

# pip install required packages
RUN pip install seaborn thop

RUN cd / && \
git clone https://github.com/JunnYu/mish-cuda && \
cd mish-cuda && \
python setup.py build install

# install pytorch_wavelets if you want to use dwt down-sampling module
# https://github.com/fbcotter/pytorch_wavelets
RUN cd / && \
git clone https://github.com/fbcotter/pytorch_wavelets && \
cd pytorch_wavelets && \
pip install .

RUN mkdir /yolor
WORKDIR /yolor
COPY ./ .
RUN pip install -qr requirements.txt
RUN cd mish-cuda && python setup.py build install
RUN cd pytorch_wavelets && pip install .