# Usa la imagen base de PyTorch con CUDA
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

# Establece la variable de entorno para desactivar el prompt de confirmación de pip
ENV PIP_NO_CACHE_DIR=1

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /neoantigenweb

# Instala las dependencias necesarias
RUN pip install transformers==4.24.0 evaluate datasets requests && \
    pip install tape-proteins && \
    pip uninstall -y protobuf && \
    pip install "protobuf==3.19.*" && \
    pip install wandb && \
    pip install scikit-learn && \
    pip uninstall -y numpy && \
    pip install numpy==1.23.5 && \
    pip install Flask

# Expone el puerto 5000 para la aplicación Flask
EXPOSE 5000
