FROM oven/bun:debian

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean

RUN apt-get install -y vim aptitude python3-pip magic-wormhole file

RUN mkdir /transformer
WORKDIR /transformer
COPY thin-plate-spline-transformer /transformer
RUN pip3 install -r requirements.txt

RUN mkdir /application
COPY nextjs /application/nextjs
WORKDIR /application/nextjs
RUN bun install
RUN bun prisma generate
ENTRYPOINT ["bun", "--bun", "next", "dev"]