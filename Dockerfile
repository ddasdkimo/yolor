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

# 針對3090支援問題修改為預覽版
RUN pip uninstall -y torch torchvision
RUN pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html

# docker build -t raidavid/yolor:0414 .
# 
# docker stop yolor_train && docker rm yolor_train &&\
# docker run  -it \
# --gpus all \
#  -d \
# -v /dev/shm:/dev/shm     \
# --memory 8000m     \
# --log-opt max-size=10m     \
# --log-opt max-file=10     \
# --name yolor_train \
# -v /home/ubuntu/tmp/test123:/yolor/data/test123 \
# raidavid/yolor:0414 \
# bash -c "python3 train.py --batch-size 1 --img 1280 1280 --data /yolor/data/test123/test123.yaml --cfg /yolor/data/test123/test123.cfg --weights runs/train/exp2/weights/last.pt --device 0 --name test123 --hyp hyp.scratch.1280.yaml --epochs 300"

# 未來商務展
# docker build -t raidavid/futurecommerce_yolor .
# docker stop saas_yolor && docker rm saas_yolor && \
# docker run  -it \
# --gpus all \
#  -d \
# -v /dev/shm:/dev/shm     \
# --memory 8000m     \
# --log-opt max-size=10m     \
# --log-opt max-file=10     \
# -p 5124:6858 \
# --name saas_yolor \
# raidavid/futurecommerce_yolor \
# bash -c "python app.py --cfg cfg/yolor_p6.cfg --weights yolor_p6.pt  --device 0 --conf-thres 0.4 --iou-thres 0.5"


# 訓練用機器
# docker stop train_yolor && docker rm train_yolor && \
# docker run  -it \
# --gpus all \
#  -d \
# -v /dev/shm:/dev/shm     \
# -v /home/ubuntu/yolor/:/yolor     \
# -v /home/ubuntu/traindata:/home/ubuntu/traindata     \
# --memory 8000m     \
# --log-opt max-size=10m     \
# --log-opt max-file=10     \
# --name train_yolor \
# raidavid/futurecommerce_yolor

# in aws
# docker run  -it \
# --gpus all \
#  -d \
# -v /dev/shm:/dev/shm     \
# -v /home/ubuntu/traindata:/home/ubuntu/traindata     \
# --memory 8000m     \
# --log-opt max-size=10m     \
# --log-opt max-file=10     \
# --name train_yolor \
# raidavid/futurecommerce_yolor