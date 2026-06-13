#!/usr/bin/env python3
import redis
import time

# Connect to Redis server
client = redis.Redis(host='localhost', port=6379, db=0)

queue_name = 'detection_queue'


def estimate_time_to_empty():
    previous_length = None
    start_time = None

    while True:
        current_length = client.llen(queue_name)

        if previous_length is not None:
            elapsed_time = time.time() - start_time
            rate = (previous_length - current_length) / elapsed_time
            if rate > 0:
                estimated_seconds = current_length / rate
                hours, remainder = divmod(estimated_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(f"Estimated time to empty the queue: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
            elif current_length == 0:
                print("Queue is empty.")
                break
            else:
                print("Queue size is increasing or stable.")

        previous_length = current_length
        start_time = time.time()

        time.sleep(10)

estimate_time_to_empty()

#  ./compute_and_store_predictions.py -q | pv -la > /dev/null
