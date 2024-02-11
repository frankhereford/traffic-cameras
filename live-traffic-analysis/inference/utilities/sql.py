import pytz
import datetime
import uuid

records_to_insert = []


def prepare_detection(
    tracker_id,
    class_id,
    image_x,
    image_y,
    timestamp,
    session_id,
    longitude,
    latitude,
):
    # Convert the Unix timestamp to a datetime value
    timestamp = datetime.datetime.fromtimestamp(timestamp)

    # Convert the timestamp to Central Time
    central = pytz.timezone("America/Chicago")
    timestamp = timestamp.astimezone(central)

    record_to_insert = (
        int(tracker_id),
        int(image_x),
        int(image_y),
        timestamp,
        session_id,
        float(longitude),
        float(latitude),
        int(class_id),
    )
    records_to_insert.append(record_to_insert)


def insert_detections(db, cursor):
    insert_query = """
    INSERT INTO detections (tracker_id, image_x, image_y, timestamp, session_id, location, class_id) 
    VALUES (%s, %s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 2253), %s)
    """
    cursor.executemany(insert_query, records_to_insert)
    db.commit()
    # Clear the records list
    records_to_insert.clear()


def get_future_locations_for_trackers(cursor, session_id, tracker_ids):
    # Ensure that tracker_ids is a list
    if tracker_ids is None:
        tracker_ids = []

    results = []

    for tracker_id in tracker_ids:
        # Define the SQL query
        query = f"""
            SELECT st_x(p.future_location) as x, st_y(p.future_location) as y
            FROM predictions p
            WHERE p.session_id = {session_id} AND p.tracker_id = {tracker_id}
            ORDER BY p.timestamp DESC
            LIMIT 1;
        """

        # Execute the query and fetch the result
        cursor.execute(query)
        result = cursor.fetchone()

        # If a result was found, add it to the results list, otherwise add None
        if result is not None:
            results.append(result)
        else:
            results.append(None)

    return results


def create_new_session(cursor):
    # Generate a fresh UUID
    new_uuid = uuid.uuid4()

    # Insert a new session record and return the id
    insert_query = """INSERT INTO sessions (uuid) VALUES (%s) RETURNING id;"""
    cursor.execute(insert_query, (str(new_uuid),))

    # Fetch the id of the newly inserted record
    session_id = cursor.fetchone()
    return session_id["id"]


def compute_speed(cursor, session_id, tracker_id, frame_look_back=15):
    cursor.execute(
        """
        WITH ranked_detections AS (
            SELECT 
                id,
                timestamp,
                ST_Transform(location, 2253) as location_transformed,
                ROW_NUMBER() OVER (ORDER BY timestamp DESC) as rn
            FROM 
                detections
            WHERE 
                session_id = %s AND 
                tracker_id = %s
        )
        SELECT 
            ABS(ST_Distance(
                (SELECT location_transformed FROM ranked_detections WHERE rn = 1),
                (SELECT location_transformed FROM ranked_detections WHERE rn = %s)
            )) as distance_in_feet,
            ABS(EXTRACT(EPOCH FROM (
                (SELECT timestamp FROM ranked_detections WHERE rn = 1) - 
                (SELECT timestamp FROM ranked_detections WHERE rn = %s)
            ))) as time_difference_in_seconds
        FROM 
            ranked_detections
        WHERE 
            rn IN (1, %s);
""",
        (
            int(session_id),
            int(tracker_id),
            frame_look_back,
            frame_look_back,
            frame_look_back,
        ),
    )
    result = cursor.fetchone()
    if (
        result
        and "distance_in_feet" in result
        and "time_difference_in_seconds" in result
        and result["distance_in_feet"] is not None
        and result["time_difference_in_seconds"] is not None
    ):
        return (
            result["distance_in_feet"] / float(result["time_difference_in_seconds"])
        ) * 0.681818
    else:
        return None


def get_class_id(db, cursor, session_id, class_id, class_name):
    # Check if the record exists
    cursor.execute(
        """
    SELECT id FROM classes 
    WHERE session_id = %s AND class_id = %s AND class_name = %s
    """,
        (session_id, int(class_id), class_name),
    )
    result = cursor.fetchone()

    # If the record exists, return its id
    if result:
        return result["id"]

    # If the record does not exist, insert it and return its id
    cursor.execute(
        """
    INSERT INTO classes (session_id, class_id, class_name) 
    VALUES (%s, %s, %s) RETURNING id
    """,
        (session_id, int(class_id), class_name),
    )
    db.commit()
    result = cursor.fetchone()
    if result:
        return result["id"]
    else:
        raise Exception("Failed to insert new record into classes table")
