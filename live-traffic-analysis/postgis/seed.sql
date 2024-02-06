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

SELECT AddGeometryColumn ('public','detections','location',2253,'POINT',2);

-- Index on session_id in classes
CREATE INDEX idx_classes_session_id ON classes(session_id);

-- Index on session_id and class_id in detections
CREATE INDEX idx_detections_session_id ON detections(session_id);
CREATE INDEX idx_detections_class_id ON detections(class_id);

-- Spatial index on the geometry column in detections
CREATE INDEX idx_detections_location ON detections USING GIST(location);

-- B-tree index on timestamp in detections, if you often filter/sort by timestamp
CREATE INDEX idx_detections_timestamp ON detections(timestamp);

-- B-tree indexes for grouping and window functions
CREATE INDEX idx_detections_session_id_tracker_id_class_id ON detections(session_id, tracker_id, class_id);
CREATE INDEX idx_classes_id_class_name ON classes(id, class_name);

-- Compound index on session_id and tracker_id in detections for specific query pattern
CREATE INDEX idx_detections_session_id_tracker_id ON detections(session_id, tracker_id);



CREATE OR REPLACE VIEW tracked_paths AS
SELECT
  MIN(detections.id) AS id,
  MIN(detections.timestamp) AS start_time,
  MAX(detections.timestamp) AS end_time,
  detections.session_id,
  detections.tracker_id,
  detections.class_id,
  classes.class_name,
  ST_ChaikinSmoothing(ST_Simplify(ST_SetSRID(ST_MakeLine(ST_Force2D(detections.location::geometry) ORDER BY detections.timestamp), 2253), 5),3) AS path,
  ST_Length(ST_ChaikinSmoothing(ST_Simplify(ST_SetSRID(ST_MakeLine(ST_Force2D(detections.location::geometry) ORDER BY detections.timestamp), 2253), 5),3)) AS distance,
  EXTRACT(EPOCH FROM (MAX(detections.timestamp) - MIN(detections.timestamp))) AS duration_seconds,
  CASE 
    WHEN EXTRACT(EPOCH FROM (MAX(detections.timestamp) - MIN(detections.timestamp))) > 0 then
    0.681818 *
    ST_Length(ST_ChaikinSmoothing(ST_Simplify(ST_SetSRID(ST_MakeLine(ST_Force2D(detections.location::geometry) ORDER BY detections.timestamp), 2253), 5),3))
      /
      EXTRACT(EPOCH FROM (MAX(detections.timestamp) - MIN(detections.timestamp)))
    ELSE
      0
  END AS average_speed_mph,
  CASE
    WHEN EXTRACT(EPOCH FROM (NOW() - MIN(detections.timestamp))) > (60 * 5) THEN 0
    ELSE 100 - (EXTRACT(EPOCH FROM (NOW() - MIN(detections.timestamp))) * 100 / (60 * 5))::integer
  END AS minute_transparency 
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
