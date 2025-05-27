
# City of Austin's Traffic Cameras

## What is this?

Yea, it's a personal project where I just followed my nose. 

Started as 

* a nextjs website to provide a better viewing experience for the COA cameras,
* into a method for using thin-plate spline techniques to warp an arbitrarily composed camera image to geographic coordinates (rubber-sheeting)
* learning how to do this on the gpu to make it fast
* doing it for live video
* object detection & tracking
* doing statistics with it
* making a machine learning model that was flawed but a huge learning experience

* a ton of random stuff while I just wrote whatever came to mind - it was good

and then i bailed -- I'll stand this up again one day, it's pretty ok. 


```
ALTER TABLE detections
ADD COLUMN location geography(Point, 4326) GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)) STORED;

ALTER TABLE locations
ADD COLUMN location geography(Point, 4326) GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)) STORED;
```