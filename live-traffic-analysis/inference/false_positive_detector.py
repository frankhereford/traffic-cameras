#!/usr/bin/env python

import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from sklearn.cluster import KMeans
import numpy as np

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

    cursor = db.cursor()

    print(db.get_dsn_parameters(), "\n")

    cursor.execute("SELECT version();")
    record = cursor.fetchone()
    print("You are connected to - ", record, "\n")

except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL", error)


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
    detection_count > avg_count + (stddev_count * 5)
ORDER BY 
    std_devs_above_avg DESC;
"""

cursor = db.cursor()
cursor.execute(sql)
results = cursor.fetchall()
for row in results:
    print(dict(row))
