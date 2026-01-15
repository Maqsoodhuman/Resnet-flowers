#!/usr/bin/env python3
"""
Delete SageMaker endpoint to stop billing.

Usage:
    python delete_endpoint.py
    python delete_endpoint.py --endpoint my-endpoint-name
"""

import argparse
import json
import os
import sys

import boto3


def load_endpoint_config():
    """Load endpoint config from deploy script output"""
    config_path = os.path.join(os.path.dirname(__file__), "endpoint_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return None


def delete_endpoint(endpoint_name: str, region: str = "us-east-1"):
    """Delete SageMaker endpoint and associated resources"""

    print(f"\nDeleting SageMaker Resources")
    print("=" * 50)
    print(f"Endpoint: {endpoint_name}")
    print(f"Region: {region}")
    print("=" * 50)

    sm_client = boto3.client("sagemaker", region_name=region)

    # Get endpoint config name
    try:
        endpoint_info = sm_client.describe_endpoint(EndpointName=endpoint_name)
        config_name = endpoint_info["EndpointConfigName"]
    except sm_client.exceptions.ClientError as e:
        print(f"Error: Endpoint '{endpoint_name}' not found.")
        sys.exit(1)

    # Get model name from config
    try:
        config_info = sm_client.describe_endpoint_config(EndpointConfigName=config_name)
        model_name = config_info["ProductionVariants"][0]["ModelName"]
    except Exception:
        model_name = None

    # Delete endpoint
    print(f"\n1. Deleting endpoint: {endpoint_name}")
    try:
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        print("   - Endpoint deletion initiated")
    except Exception as e:
        print(f"   - Error: {e}")

    # Delete endpoint config
    print(f"\n2. Deleting endpoint config: {config_name}")
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=config_name)
        print("   - Endpoint config deleted")
    except Exception as e:
        print(f"   - Error: {e}")

    # Delete model
    if model_name:
        print(f"\n3. Deleting model: {model_name}")
        try:
            sm_client.delete_model(ModelName=model_name)
            print("   - Model deleted")
        except Exception as e:
            print(f"   - Error: {e}")

    # Remove config file
    config_path = os.path.join(os.path.dirname(__file__), "endpoint_config.json")
    if os.path.exists(config_path):
        os.remove(config_path)
        print("\n4. Removed local endpoint_config.json")

    print("\n" + "=" * 50)
    print("CLEANUP COMPLETE!")
    print("=" * 50)
    print("\nNote: S3 model artifacts remain for future deployments.")
    print("To fully clean up, delete the 'flower-classifier' prefix in your S3 bucket.\n")


def main():
    parser = argparse.ArgumentParser(description="Delete SageMaker Flower Classification Endpoint")
    parser.add_argument("--endpoint", "-e", help="SageMaker endpoint name")
    parser.add_argument("--region", "-r", default="us-east-1", help="AWS region (default: us-east-1)")

    args = parser.parse_args()

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
            print("Use --endpoint to specify the endpoint name to delete.")
            sys.exit(1)

    # Confirm deletion
    confirm = input(f"\nAre you sure you want to delete endpoint '{endpoint_name}'? (yes/no): ")
    if confirm.lower() != "yes":
        print("Deletion cancelled.")
        sys.exit(0)

    delete_endpoint(endpoint_name, region)


if __name__ == "__main__":
    main()
