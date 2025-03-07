# Change the base image according to your setup, view here other versions:
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html
FROM nvcr.io/nvidia/pytorch:21.04-py3

COPY . /opt/CM4IR

WORKDIR /opt/CM4IR

# Other packages in requirements.txt alraedy exists in base image
RUN pip install lpips einops==0.8.0 flash-attn==0.2.8 numpy==1.24.4

#CMD ./evaluate_CM4IR.sh
CMD sleep infinity