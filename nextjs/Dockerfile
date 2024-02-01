FROM node:latest

WORKDIR /app

COPY package*.json ./

RUN npm install
RUN npx prisma generate

COPY . .

# EXPOSE 3000
EXPOSE 5555

CMD [ "npm", "run", "dev" ]
