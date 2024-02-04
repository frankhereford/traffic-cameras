FROM postgis/postgis

# Set the timezone
ENV TZ=America/Chicago

# Install tzdata package and set timezone
RUN apt-get update && apt-get install -y tzdata && \
  ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone