#!/usr/bin/env python3
"""
Test client for the deployed SageMaker Flower Classification endpoint.

Usage:
    python test_endpoint.py --image path/to/flower.jpg
    python test_endpoint.py --image path/to/flower.jpg --endpoint my-endpoint-name

Alternative using AWS CLI:
    aws sagemaker-runtime invoke-endpoint \
        --endpoint-name <endpoint-name> \
        --content-type image/jpeg \
        --body fileb://path/to/image.jpg \
        output.json
"""

import argparse
import json
import os
import sys

try:
    import boto3
except ImportError:
    print("boto3 not found. Use AWS CLI instead:")
    print("  aws sagemaker-runtime invoke-endpoint --endpoint-name <name> --content-type image/jpeg --body fileb://image.jpg output.json")
    sys.exit(1)


def load_endpoint_config():
    """Load endpoint config from deploy script output"""
    config_path = os.path.join(os.path.dirname(__file__), "endpoint_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return None


def test_endpoint(image_path: str, endpoint_name: str, region: str = "us-east-1"):
    """Send an image to the endpoint and get predictions"""

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    print(f"\nTesting SageMaker Endpoint")
    print("=" * 50)
    print(f"Endpoint: {endpoint_name}")
    print(f"Region: {region}")
    print(f"Image: {image_path}")
    print("=" * 50)

    # Read image bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Create SageMaker runtime client
    runtime = boto3.client("sagemaker-runtime", region_name=region)

    print("\nSending request...")

    # Invoke endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="image/jpeg",
        Accept="application/json",
        Body=image_bytes
    )

    # Parse response
    result = json.loads(response["Body"].read().decode())

    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"\nPredicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")

    print("\nTop 5 Predictions:")
    print("-" * 40)
    for i, pred in enumerate(result["predictions"], 1):
        print(f"  {i}. Class {pred['class_name']}: {pred['probability']*100:.2f}%")

    print("=" * 50 + "\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Test SageMaker Flower Classification Endpoint")
    parser.add_argument("--image", "-i", required=True, help="Path to flower image (JPEG/PNG)")
    parser.add_argument("--endpoint", "-e", help="SageMaker endpoint name")
    parser.add_argument("--region", "-r", default="us-east-1", help="AWS region (default: us-east-1)")

    args = parser.parse_args()

    # Get endpoint name
    endpoint_name = args.endpoint
    region = args.region

    if not endpoint_name:
        config = load_endpoint_config()
        if config:
            endpoint_name = config.get("endpoint_name")
            region = config.get("region", region)
            print(f"Using endpoint from config: {endpoint_name}")
        else:
            print("Error: No endpoint specified and no config found.")
            print("Use --endpoint to specify endpoint name, or run deploy.py first.")
            sys.exit(1)

    test_endpoint(args.image, endpoint_name, region)


if __name__ == "__main__":
    main()
