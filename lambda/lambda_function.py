import sys
import json

def handler(event, context):
    result = {
        "event": event,
        "context": str(context)
    }
    return {
        "statusCode": 200,
        "body": json.dumps(result)
    }