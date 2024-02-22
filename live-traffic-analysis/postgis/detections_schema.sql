CREATE SCHEMA detections;

drop table if exists detections.recordings cascade;
drop table if exists detections.classes cascade;
drop table if exists detections.frames cascade;
drop table if exists detections.trackers cascade;
drop table if exists detections.detections cascade;

CREATE TABLE detections.recordings (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL
);


CREATE TABLE detections.classes (
    id SERIAL PRIMARY KEY,
    recording_id INT NOT NULL,
    ultralytics_id INT NOT NULL,
    ultralytics_name TEXT NOT NULL,
    FOREIGN KEY (recording_id) REFERENCES detections.recordings(id)
);


CREATE TABLE detections.frames (
    id SERIAL PRIMARY KEY,
    recording_id INT NOT NULL,
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    hash TEXT NOT NULL,
    FOREIGN KEY (recording_id) REFERENCES detections.recordings(id)
);


CREATE TABLE detections.trackers (
    id SERIAL PRIMARY KEY,
    class_id INT NOT NULL,
    ultralytics_id INT NOT null,
    FOREIGN KEY (class_id) REFERENCES detections.classes(id)
);

CREATE TABLE detections.detections (
    id SERIAL PRIMARY KEY,
    frame_id INT NOT NULL,
    tracker_id INT NOT NULL,
    x1 FLOAT NOT NULL,
    y1 FLOAT NOT NULL,
    x2 FLOAT NOT NULL,
    y2 FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    location public.geometry(Point,2253),
    FOREIGN KEY (frame_id) REFERENCES detections.frames(id),
    FOREIGN KEY (tracker_id) REFERENCES detections.trackers(id)
);





-- Indexes for detections.recordings
CREATE INDEX idx_recordings_start_time ON detections.recordings (start_time);
-- Important if queries often filter or sort by start_time.

-- Indexes for detections.classes
CREATE INDEX idx_classes_recording_id ON detections.classes (recording_id);
-- Crucial for JOIN performance with the recordings table.
CREATE INDEX idx_classes_ultralytics_id ON detections.classes (ultralytics_id);
-- Useful if ultralytics_id is frequently used in queries.

-- Indexes for detections.frames
CREATE INDEX idx_frames_recording_id ON detections.frames (recording_id);
-- Essential for JOIN operations with the recordings table.
CREATE INDEX idx_frames_time ON detections.frames (time);
-- Important if queries often filter or sort by time.
CREATE INDEX idx_frames_hash ON detections.frames (hash);
-- Beneficial for searching or deduplicating based on hash.

-- Indexes for detections.trackers
CREATE INDEX idx_trackers_class_id ON detections.trackers (class_id);
-- Important for JOINs with the classes table.
CREATE INDEX idx_trackers_ultralytics_id ON detections.trackers (ultralytics_id);
-- Helpful if ultralytics_id is commonly queried.

-- Indexes for detections.detections
CREATE INDEX idx_detections_frame_id ON detections.detections (frame_id);
-- Vital for JOINs with the frames table.
CREATE INDEX idx_detections_tracker_id ON detections.detections (tracker_id);
-- Important for JOINs with the trackers table.
CREATE INDEX idx_detections_confidence ON detections.detections (confidence);
-- Useful if confidence is a common filter in queries.
CREATE INDEX idx_detections_location ON detections.detections USING gist(location);
-- Crucial for spatial queries involving the location column.

