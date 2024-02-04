CREATE TABLE sessions (
    id SERIAL PRIMARY KEY,
    uuid TEXT,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE classes (
    id SERIAL PRIMARY KEY,
    session_id int not null REFERENCES sessions(id) ON DELETE CASCADE,
    class_id int not null,
    class_name TEXT not null,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE detections (
    id SERIAL PRIMARY KEY,
    timestamp timestamptz NOT null,
    session_id int not null REFERENCES sessions(id) ON DELETE CASCADE,
    tracker_id int not null,
    class_id int not null REFERENCES classes(id) ON DELETE CASCADE,
    image_x FLOAT not null,
    image_y FLOAT not null
);

SELECT AddGeometryColumn ('public','detections','location',4326,'POINT',2);


CREATE OR REPLACE VIEW tracked_paths AS
SELECT
  MIN(detections.id) AS id,
  MIN(detections.timestamp) AS start_time,
  MAX(detections.timestamp) AS end_time,
  detections.session_id,
  detections.tracker_id,
  detections.class_id,
  classes.class_name,
  ST_SetSRID(ST_MakeLine(ST_Force2D(detections.location::geometry) ORDER BY detections.timestamp), 4326) AS path,
  ST_Length(ST_Transform(ST_SetSRID(ST_MakeLine(ST_Force2D(detections.location::geometry) ORDER BY detections.timestamp), 4326), 2229)) AS distance,
  EXTRACT(EPOCH FROM (MAX(detections.timestamp) - MIN(detections.timestamp))) AS duration_seconds,
  CASE 
    WHEN EXTRACT(EPOCH FROM (MAX(detections.timestamp) - MIN(detections.timestamp))) > 0 THEN
    ST_Length(ST_Transform(ST_SetSRID(ST_MakeLine(ST_Force2D(detections.location::geometry) ORDER BY detections.timestamp), 4326), 2229))
      /
      EXTRACT(EPOCH FROM (MAX(detections.timestamp) - MIN(detections.timestamp)))
    ELSE
      0
  END AS average_speed_fps 
FROM
  detections
  JOIN classes ON detections.class_id = classes.id
GROUP BY
  detections.session_id, detections.tracker_id, detections.class_id, classes.class_name 
HAVING
  COUNT(detections.id) >= 5 AND
  ST_MakeLine(ST_Force2D(detections.location::geometry) ORDER BY detections.timestamp) IS NOT NULL
ORDER BY
  session_id, tracker_id;