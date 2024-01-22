#docker run --platform linux/amd64 -it cnstark/pytorch:2.0.1-py3.10.11-ubuntu22.04 bash
# Dockerfile-python
FROM cnstark/pytorch:2.0.1-py3.10.11-ubuntu22.04

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY load_model.py ./
RUN python ./load_model.py

COPY . .

EXPOSE 5000

CMD [ "python", "app.py" ]

