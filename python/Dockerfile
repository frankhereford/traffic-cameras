FROM cnstark/pytorch:2.0.1-py3.10.11-ubuntu22.04

RUN apt-get update && \
    apt-get install -y curl fontconfig wget vim aptitude gdal-bin

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

RUN npm install prisma --save-dev

WORKDIR /usr/share/fonts/truetype/

RUN wget https://github.com/githubnext/monaspace/raw/main/fonts/otf/MonaspaceNeon-Light.otf

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN pip install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

COPY load_model.py ./
RUN python ./load_model.py

COPY . .

RUN prisma generate

EXPOSE 5000

CMD [ "python", "app.py", "--mode=flask"]

