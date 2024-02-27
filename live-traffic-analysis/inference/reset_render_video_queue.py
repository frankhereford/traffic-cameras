#!/usr/bin/env python3

import os
import redis

# Connect to Redis
r = redis.Redis(host="localhost", port=6379, db=0)

r.flushall()

# List all .mp4 files in ./input_media
files = [f for f in os.listdir("./input_media") if f.endswith(".mp4")]

# Sort the files in alphabetical order
files.sort()

# Add the files to the Redis queue
for file in files:
    r.lpush("render-videos-queue", file)
