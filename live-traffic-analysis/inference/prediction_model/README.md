# Prediction Model

## Preparation

- Refresh `detections.detections_on_short_tracks` after setting the minimum track length.

```sql
create materialized view detections.detections_on_short_tracks as (
with short_tracks as (
select
  min(frames.time) as start_time,
  max(frames.time) as end_time,
  EXTRACT(epoch FROM max(frames.time) - min(frames.time)) AS duration_seconds,
  min(recordings.id) as recording_id,
  min(trackers.id) as tracker_id,
  min(classes.ultralytics_name) as class_name,
  count(detections.id) as total_detections,
  st_length(st_chaikinsmoothing(st_simplify(st_setsrid(ST_MakeLine(ST_Force2D(detections.location::geometry) ORDER BY frames.time), 2253),5), 3)) as length_feet
from detections.trackers trackers
join detections.detections detections on (detections.tracker_id = trackers.id)
join detections.frames frames on (detections.frame_id = frames.id)
join detections.recordings recordings on (frames.recording_id = recordings.id)
join detections.classes classes on (trackers.class_id = classes.id)
group by trackers.id
having st_length(st_chaikinsmoothing(st_simplify(st_setsrid(ST_MakeLine(ST_Force2D(detections.location::geometry) ORDER BY frames.time), 2253),5), 3)) < 10 -- feet
)
SELECT
    tracks.tracker_id,
    detections.id as detections_id,
    frames.id as frame_id,
    frames."time" as frame_time,
    detections.x1,
    detections.y1,
    detections.x2,
    detections.y2,
    (detections.x1 + detections.x2) / 2 AS centerX,
    (detections.y1 + detections.y2) / 2 AS centerY,
    detections."location" as local_location,
    ST_Transform(detections."location",4326) as wgs84_location
FROM
    short_tracks tracks
JOIN detections.detections detections ON (tracks.tracker_id = detections.tracker_id)
join detections.frames frames on (detections.frame_id = frames.id)
order by tracks.tracker_id asc, frames.time asc
);

CREATE UNIQUE INDEX idx_detections_id ON detections.detections_on_short_tracks (detections_id);
```

- Mark trackers which were too short.

```sql
UPDATE detections.trackers set discard_short_track = FALSE;
UPDATE detections.trackers
SET discard_short_track = TRUE
WHERE id IN (
    SELECT DISTINCT tracks.tracker_id
    FROM detections.detections_on_short_tracks tracks
);
```

- Update materialized view `detections.tracks` to have enough minimum number of points for each track. The minimum is the sum of the input and the predictive sizes. 1 second of input = 30 tracks, and 1 second of prediction = 30, so 60.

```sql
CREATE MATERIALIZED VIEW detections.tracks
TABLESPACE pg_default
AS SELECT min(frames."time") AS start_time,
    max(frames."time") AS end_time,
    EXTRACT(epoch FROM max(frames."time") - min(frames."time")) AS duration_seconds,
    min(recordings.start_time) AS recording_start_time,
    min(recordings.id) AS recording_id,
    min(trackers.id) AS tracker_id,
    min(classes.ultralytics_name) AS class_name,
    count(detections.id) AS total_detections,
    st_length(st_chaikinsmoothing(st_simplify(st_setsrid(st_makeline(st_force2d(detections.location::geometry) ORDER BY frames."time"), 2253), 5::double precision), 3)) / 5280::double precision / (EXTRACT(epoch FROM max(frames."time") - min(frames."time")) / 3600::numeric)::double precision AS rate_mph,
    st_transform(st_chaikinsmoothing(st_simplify(st_setsrid(st_makeline(st_force2d(detections.location::geometry) ORDER BY frames."time"), 2253), 5::double precision), 3), 4326) AS wgs84_track,
    st_chaikinsmoothing(st_simplify(st_setsrid(st_makeline(st_force2d(detections.location::geometry) ORDER BY frames."time"), 2253), 5::double precision), 3) AS local_projection_track,
    st_length(st_chaikinsmoothing(st_simplify(st_setsrid(st_makeline(st_force2d(detections.location::geometry) ORDER BY frames."time"), 2253), 5::double precision), 3)) AS length_feet
   FROM detections.trackers trackers
     JOIN detections.detections detections ON detections.tracker_id = trackers.id
     JOIN detections.frames frames ON detections.frame_id = frames.id
     JOIN detections.recordings recordings ON frames.recording_id = recordings.id
     JOIN detections.classes classes ON trackers.class_id = classes.id
  GROUP BY trackers.id
 HAVING trackers.discard_short_track = false AND count(detections.id) > 60 AND (st_length(st_chaikinsmoothing(st_simplify(st_setsrid(st_makeline(st_force2d(detections.location::geometry) ORDER BY frames."time"), 2253), 5::double precision), 3)) / 5280::double precision / (EXTRACT(epoch FROM max(frames."time") - min(frames."time")) / 3600::numeric)::double precision) > 5::double precision
  ORDER BY (count(detections.id)) DESC
WITH DATA;
```

- Empty out your training data tables

```sql
truncate training_data.samples cascade;
```

- Populate those tables with the python script `live-traffic-analysis/inference/populate_training_data_schema.py`

- Truncate and populate the `training_data.sample_data` table

```sql
TRUNCATE TABLE training_data.sample_data;

INSERT INTO training_data.sample_data (sample_id, detection_ids, coordinates)
    SELECT
    training_data.samples.id AS sample_id,
    ARRAY_AGG(detections.detections.id ORDER BY frames.time) AS detection_ids,
    ARRAY_AGG(ARRAY[ST_X(detections.detections.location), ST_Y(detections.detections.location)] ORDER BY frames.time) AS coordinates
FROM
    training_data.samples
JOIN
    training_data.detections ON training_data.samples.id = training_data.detections.sample_id
JOIN
    detections.detections ON training_data.detections.detection_id = detections.detections.id
JOIN
    detections.frames ON detections.detections.frame_id = detections.frames.id
GROUP BY
    training_data.samples.id;
```
