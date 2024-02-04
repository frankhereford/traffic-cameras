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

CREATE OR REPLACE FUNCTION CreateCurve(geom geometry, percent int DEFAULT 40)
    RETURNS geometry AS
$$
DECLARE
    result text;
    p0 geometry;
    p1 geometry;
    p2 geometry;
    intp geometry;
    tempp geometry;
    geomtype text := ST_GeometryType(geom);
    factor double precision := percent::double precision / 200;
    i integer;
BEGIN
    IF percent < 0 OR percent > 100 THEN
        RAISE EXCEPTION 'Smoothing factor must be between 0 and 100';
    END IF;
    IF geomtype != 'ST_LineString' OR factor = 0 THEN
        RETURN geom;
    END IF;
    result := 'COMPOUNDCURVE((';
    p0 := ST_PointN(geom, 1);
    IF ST_NPoints(geom) = 2 THEN
        p1:= ST_PointN(geom, 2);
        result := result || ST_X(p0) || ' ' || ST_Y(p0) || ',' || ST_X(p1) || ' ' || ST_Y(p1) || '))';
    ELSE
        FOR i IN 2..(ST_NPoints(geom) - 1) LOOP
            p1 := ST_PointN(geom, i);
            p2 := ST_PointN(geom, i + 1);
            result := result || ST_X(p0) || ' ' || ST_Y(p0) || ',';
            tempp := ST_LineInterpolatePoint(ST_MakeLine(p1, p0), factor);
            p0 := ST_LineInterpolatePoint(ST_MakeLine(p1, p2), factor);
            intp := ST_LineInterpolatePoint(
                ST_MakeLine(
                    ST_LineInterpolatePoint(ST_MakeLine(p0, p1), 0.5),
                    ST_LineInterpolatePoint(ST_MakeLine(tempp, p1), 0.5)
                ), 0.5);
            result := result || ST_X(tempp) || ' ' || ST_Y(tempp) || '),CIRCULARSTRING(' || ST_X(tempp) || ' ' || ST_Y(tempp) || ',' || ST_X(intp) || ' ' ||
            ST_Y(intp) || ',' || ST_X(p0) || ' ' || ST_Y(p0) || '),(';
        END LOOP;
        result := result || ST_X(p0) || ' ' || ST_Y(p0) || ',' || ST_X(p2) || ' ' || ST_Y(p2) || '))';
    END IF;
    RETURN ST_SetSRID(result::geometry, ST_SRID(geom));
END;
$$
LANGUAGE 'plpgsql' IMMUTABLE;


CREATE OR REPLACE VIEW tracked_paths AS
SELECT
  MIN(detections.id) AS id,
  MIN(detections.timestamp) AS start_time,
  MAX(detections.timestamp) AS end_time,
  detections.session_id,
  detections.tracker_id,
  detections.class_id,
  classes.class_name,
  ST_ChaikinSmoothing(ST_Simplify(ST_Transform(ST_SetSRID(ST_MakeLine(ST_Force2D(detections.location::geometry) ORDER BY detections.timestamp), 4326), 2229), 5),3) AS path,
  ST_Length(ST_ChaikinSmoothing(ST_Simplify(ST_Transform(ST_SetSRID(ST_MakeLine(ST_Force2D(detections.location::geometry) ORDER BY detections.timestamp), 4326), 2229), 5),3)) AS distance,
  EXTRACT(EPOCH FROM (MAX(detections.timestamp) - MIN(detections.timestamp))) AS duration_seconds,
  CASE 
    WHEN EXTRACT(EPOCH FROM (MAX(detections.timestamp) - MIN(detections.timestamp))) > 0 then
    0.681818 *
    ST_Length(ST_ChaikinSmoothing(ST_Simplify(ST_Transform(ST_SetSRID(ST_MakeLine(ST_Force2D(detections.location::geometry) ORDER BY detections.timestamp), 4326), 2229), 5),3))
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