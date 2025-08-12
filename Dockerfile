# TensorFlow з GPU підтримкою (CUDA 11.8 + cuDNN 8.6)
FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

RUN pip install --upgrade pip    
RUN pip install matplotlib jupyter
RUN pip install slugify
RUN pip install "numpy<2.0"


# Додаткові порти (за потреби)
EXPOSE 8888

# CMD [ "python3", "fish.py" ]
