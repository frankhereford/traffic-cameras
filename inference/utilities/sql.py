import pytz
import datetime
import uuid


def insert_detection(
    db,
    cursor,
    tracker_id,
    class_id,
    image_x,
    image_y,
    timestamp,
    session_id,
    longitude,
    latitude,
):

    insert_query = """
    INSERT INTO detections (tracker_id, image_x, image_y, timestamp, session_id, location, class_id) 
    VALUES (%s, %s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s)
    """
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
    cursor.execute(insert_query, record_to_insert)
    db.commit()


def create_new_session(cursor):
    # Generate a fresh UUID
    new_uuid = uuid.uuid4()

    # Insert a new session record and return the id
    insert_query = """INSERT INTO sessions (uuid) VALUES (%s) RETURNING id;"""
    cursor.execute(insert_query, (str(new_uuid),))

    # Fetch the id of the newly inserted record
    session_id = cursor.fetchone()
    return session_id["id"]


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
