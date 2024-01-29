# City of Austin's Traffic Cameras

## Features

- Object detection using the `facebook/detr-resnet-101` model courtesy of [Hugging Face](https://huggingface.co/facebook/detr-resnet-101)
- [Thin Plate Spline](https://en.wikipedia.org/wiki/Thin_plate_spline) transformation to correct for camera perspective
- First class geospatial data via postGIS

## Database

### Computed fields for proper geometry

```sql
ALTER TABLE detections
ADD COLUMN location geography(Point, 4326) GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)) STORED;

ALTER TABLE locations
ADD COLUMN location geography(Point, 4326) GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)) STORED;
```
