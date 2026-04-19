FROM node:latest

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

RUN npx prisma generate

# EXPOSE 3000
EXPOSE 5555

CMD [ "npm", "run", "dev" ]
