#!/bin/bash
set -e

# Configuration
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
STACK_NAME="flower-classifier-${TIMESTAMP}"
BUCKET_NAME="sagemaker-${REGION}-${ACCOUNT_ID}"
MODEL_PREFIX="flower-classifier/models"

echo "=============================================="
echo "  ResNet-34 Flower Classifier - AWS Deployment"
echo "=============================================="
echo "Account: ${ACCOUNT_ID}"
echo "Region: ${REGION}"
echo "Stack: ${STACK_NAME}"
echo ""

# Step 1: Create S3 bucket if it doesn't exist
echo "Step 1: Checking S3 bucket..."
if aws s3 ls "s3://${BUCKET_NAME}" 2>&1 | grep -q 'NoSuchBucket'; then
    echo "  Creating bucket: ${BUCKET_NAME}"
    aws s3 mb "s3://${BUCKET_NAME}" --region ${REGION}
else
    echo "  Bucket exists: ${BUCKET_NAME}"
fi

# Step 2: Package model
echo ""
echo "Step 2: Packaging model..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TEMP_DIR="${SCRIPT_DIR}/temp_model"
MODEL_TAR="${SCRIPT_DIR}/model.tar.gz"

rm -rf "${TEMP_DIR}"
mkdir -p "${TEMP_DIR}"

# Copy weights
cp "${PROJECT_DIR}/resnet34_flowers_best.pth" "${TEMP_DIR}/"

# Copy inference code
cp -r "${SCRIPT_DIR}/code" "${TEMP_DIR}/"

# Create tar.gz
cd "${TEMP_DIR}"
tar -czvf "${MODEL_TAR}" . > /dev/null
cd "${SCRIPT_DIR}"
rm -rf "${TEMP_DIR}"

MODEL_SIZE=$(ls -lh "${MODEL_TAR}" | awk '{print $5}')
echo "  Model package: ${MODEL_SIZE}"

# Step 3: Upload to S3
echo ""
echo "Step 3: Uploading to S3..."
S3_MODEL_PATH="s3://${BUCKET_NAME}/${MODEL_PREFIX}/model.tar.gz"
aws s3 cp "${MODEL_TAR}" "${S3_MODEL_PATH}"
echo "  Uploaded: ${S3_MODEL_PATH}"

# Step 4: Deploy CloudFormation stack
echo ""
echo "Step 4: Deploying CloudFormation stack..."
aws cloudformation deploy \
    --template-file "${SCRIPT_DIR}/cloudformation.yaml" \
    --stack-name "${STACK_NAME}" \
    --parameter-overrides \
        ModelDataUrl="${S3_MODEL_PATH}" \
        BucketName="${BUCKET_NAME}" \
    --capabilities CAPABILITY_NAMED_IAM \
    --region ${REGION}

# Get outputs
echo ""
echo "Step 5: Getting endpoint details..."
ENDPOINT_NAME=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --query "Stacks[0].Outputs[?OutputKey=='EndpointName'].OutputValue" \
    --output text \
    --region ${REGION})

# Save config
cat > "${SCRIPT_DIR}/endpoint_config.json" << EOF
{
    "endpoint_name": "${ENDPOINT_NAME}",
    "region": "${REGION}",
    "stack_name": "${STACK_NAME}",
    "s3_model_path": "${S3_MODEL_PATH}"
}
EOF

echo ""
echo "=============================================="
echo "  DEPLOYMENT COMPLETE!"
echo "=============================================="
echo ""
echo "  Endpoint Name: ${ENDPOINT_NAME}"
echo "  Region: ${REGION}"
echo "  Stack: ${STACK_NAME}"
echo ""
echo "  Test with:"
echo "    python test_endpoint.py --image ../flowers/test/001/image_00001.jpg"
echo ""
echo "  Delete with:"
echo "    aws cloudformation delete-stack --stack-name ${STACK_NAME}"
echo "=============================================="
