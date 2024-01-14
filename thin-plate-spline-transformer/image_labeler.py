import argparse
import boto3
import json


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("uuid", help="The UUID to process")
    args = parser.parse_args()

    # Read data from file
    with open(f"/tmp/transformations/{args.uuid}/cctvImage.jpg", "rb") as image:
        image_bytes = image.read()

    # Create a Rekognition client
    rekognition = boto3.client("rekognition")

    # Submit the image to AWS Rekognition and get back labels
    response = rekognition.detect_labels(Image={"Bytes": image_bytes})

    # Output the response data as JSON
    print(json.dumps(response))  # Modify this line

    # Print the labels
    # for label in response["Labels"]:
    #     print(label["Name"])


if __name__ == "__main__":
    main()
