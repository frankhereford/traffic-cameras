import sys
import json

def handler(event, context):
    """
    Main Lambda handler function
    Parameters:
        event: Dict containing the Lambda function event data
        context: Lambda runtime context
    Returns:
        Dict containing status message
    """
    try:
        # Log the event for debugging
        print("Received event: " + json.dumps(event))
        # print(f"The name: {event["name"]}")

        input = {
            'coaId': event["coaId"],
            'hash': event["hash"],
        }

        # Process the message (for example, just return it)
        response = {
            'statusCode': 200,
            'body': json.dumps(input)
        }

        return response

    except Exception as e:
        # Handle any exceptions that occur
        print(f"Error processing event: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
    

