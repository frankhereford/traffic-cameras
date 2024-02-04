#!/bin/bash

docker compose stop live_camera_postgis;
docker container prune;
docker volume rm -y traffic-cameras_db_live_camera_data;
docker compose up live_camera_postgis -d;

