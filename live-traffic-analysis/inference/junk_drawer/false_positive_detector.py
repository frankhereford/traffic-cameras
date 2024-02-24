#!/usr/bin/env python

import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from sklearn.cluster import KMeans
import numpy as np
import subprocess
import ffmpeg
from PIL import Image, ImageDraw


load_dotenv()

try:
    db = psycopg2.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT"),
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        cursor_factory=psycopg2.extras.RealDictCursor,
    )

    # cursor = db.cursor()

    # print(db.get_dsn_parameters(), "\n")

    # cursor.execute("SELECT version();")
    # record = cursor.fetchone()
    # print("You are connected to - ", record, "\n")

except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL", error)


def get_single_hls_frame_as_image(hls_url, fps=30):
    # Set up the ffmpeg command to capture the stream
    command = (
        ffmpeg.input(hls_url, format="hls", loglevel="quiet", vcodec="h264_cuvid")
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", r=fps)
        .global_args("-re")
        .compile()
    )
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Read 1920*1080*3 bytes (= 1 frame)
    in_bytes = process.stdout.read(1920 * 1080 * 3)
    process.terminate()

    if not in_bytes:
        return None  # or handle the error as you prefer

    frame = np.frombuffer(in_bytes, np.uint8).reshape([1080, 1920, 3])

    # Convert to a Pillow Image
    image = Image.fromarray(frame)
    return image


sql = """
WITH DetectionStats AS (
    SELECT
        AVG(detection_count) AS avg_count,
        STDDEV(detection_count) AS stddev_count
    FROM (
        SELECT 
            COUNT(*) as detection_count
        FROM 
            detections
        GROUP BY 
            image_x, image_y
    ) AS SubQuery
),
OutlierDetections AS (
    SELECT 
        image_x, 
        image_y, 
        COUNT(*) as detection_count
    FROM 
        detections
    GROUP BY 
        image_x, image_y
)
SELECT 
    image_x::integer, 
    image_y::integer, 
    detection_count,
    round((detection_count - avg_count) / stddev_count,0)::integer AS std_devs_above_avg
FROM 
    OutlierDetections, DetectionStats
WHERE 
    detection_count > avg_count + (stddev_count * 15)
ORDER BY 
    std_devs_above_avg DESC;
"""

cursor = db.cursor()
cursor.execute(sql)
results = cursor.fetchall()
transformed_results = [(row["image_x"], row["image_y"], 10) for row in results]
print("transformed_results: ", transformed_results)


data = np.array([[row["image_x"], row["image_y"]] for row in results])

print("results length: ", len(results))

# Assuming a certain number of clusters for demonstration, this can be adjusted
k = 6
kmeans = KMeans(n_clusters=k)
kmeans.fit(data)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_


# Calculating the average radius for each cluster
clusters = {i: data[np.where(labels == i)] for i in range(k)}
average_radii = {}
for i, points in clusters.items():
    dist = np.sqrt(np.sum((points - centroids[i]) ** 2, axis=1))
    average_radii[i] = np.mean(dist)

# print(centroids)
# print(average_radii)

max_distance = 30
min_distance = 4

cluster_data = [
    (round(x), round(y), max(round(distance), min_distance))
    for (x, y), distance in zip(centroids, average_radii.values())
]

# print("clusters: ", cluster_data)

hls_url = "http://10.0.3.228:8080/memfs/9ea806cb-a214-4971-8b29-76cc9fc9de75.m3u8"
frame = get_single_hls_frame_as_image(hls_url)


def annotate_image_clusters(image, clusters):
    draw = ImageDraw.Draw(image)
    for x, y, radius in clusters:
        # Calculate the top-left and bottom-right points for the bounding box
        top_left = (x - radius, y - radius)
        bottom_right = (x + radius, y + radius)

        # Draw a circle for each cluster
        draw.ellipse([top_left, bottom_right], outline="red", width=2)
    return image


def annotate_image_with_dots(image, results):
    draw = ImageDraw.Draw(image)
    for row in results:
        x = row["image_x"]
        y = row["image_y"]
        radius = 5  # Small fixed radius for the dot

        # Calculate the top-left and bottom-right points for the bounding box
        top_left = (x - radius, y - radius)
        bottom_right = (x + radius, y + radius)

        # Draw a green dot for each point
        draw.ellipse([top_left, bottom_right], fill="green", outline="green")

    return image


annotated_image = annotate_image_with_dots(frame, results)
# annotated_image = annotate_image_clusters(annotated_image, cluster_data)


# Save the result
annotated_image.save("k-means-cluster-results.jpg")
