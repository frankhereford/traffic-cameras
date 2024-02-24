SELECT 
    detections.session_id, 
    detections.tracker_id, 
    ARRAY_AGG(ST_X(detections.location) ORDER BY detections.timestamp) as x_coords,
    ARRAY_AGG(ST_Y(detections.location) ORDER BY detections.timestamp) as y_coords,
    ARRAY_AGG(EXTRACT(EPOCH FROM detections.timestamp) ORDER BY detections.timestamp) as timestamps,
    MIN(detections.timestamp) as start_timestamp,
    paths.distance as track_length
FROM 
    detections_extended detections
    LEFT JOIN tracked_paths paths ON (detections.session_id = paths.session_id AND detections.tracker_id = paths.tracker_id)
WHERE 
    paths.distance IS NOT NULL
    AND paths.distance >= 15
GROUP BY 
    detections.session_id, detections.tracker_id, paths.distance
HAVING 
    COUNT(*) > ({SEGMENT_LENGTH} + {PREDICTION_DISTANCE} + 5)
ORDER BY 
    MIN(detections.timestamp) asc

--this query is really just an experiment; it filters down tracks to ones with significant motion
WITH ordered_detections AS (
SELECT 
    session_id, 
    tracker_id, 
    ST_X(location) as x_coord,
    ST_Y(location) as y_coord,
    ROW_NUMBER() OVER(PARTITION BY session_id, tracker_id ORDER BY timestamp) as rn
FROM 
    detections_extended
),
distances AS (
SELECT 
    d1.session_id, 
    d1.tracker_id, 
    SQRT(POWER(d1.x_coord - d2.x_coord, 2) + POWER(d1.y_coord - d2.y_coord, 2)) as distance
FROM 
    ordered_detections d1
    JOIN ordered_detections d2 ON d1.session_id = d2.session_id AND d1.tracker_id = d2.tracker_id
WHERE 
    d1.rn = 1 AND d2.rn = 30
)
SELECT 
detections.session_id, 
detections.tracker_id, 
ARRAY_AGG(ST_X(detections.location) ORDER BY detections.timestamp) as x_coords,
ARRAY_AGG(ST_Y(detections.location) ORDER BY detections.timestamp) as y_coords,
ARRAY_AGG(EXTRACT(EPOCH FROM detections.timestamp) ORDER BY detections.timestamp) as timestamps,
MIN(detections.timestamp) as start_timestamp,
paths.distance as track_length
FROM 
detections_extended detections
LEFT JOIN tracked_paths paths ON (detections.session_id = paths.session_id AND detections.tracker_id = paths.tracker_id)
JOIN distances ON (detections.session_id = distances.session_id AND detections.tracker_id = distances.tracker_id)
WHERE 
paths.distance IS NOT NULL
AND paths.distance >= 30
AND distances.distance > 30
GROUP BY 
detections.session_id, detections.tracker_id, paths.distance
HAVING 
COUNT(*) > 60
ORDER BY 
MIN(detections.timestamp) asc