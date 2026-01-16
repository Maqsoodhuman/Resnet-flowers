# ResNet-34 Flower Classification - AWS SageMaker Deployment

Development and Deployment of a trained ResNet-34 flower classification model to AWS SageMaker with a real-time inference endpoint.

## Model Details

| Attribute | Value |
|-----------|-------|
| Architecture | ResNet-34 (custom implementation) |
| Framework | PyTorch 2.0 |
| Classes | 102 flower species |
| Input | 224x224 RGB images |
| Accuracy | 93.77% (test set) |
| Parameters | 21.3M |

## Project Structure

```
.
├── code/
│   ├── model.py              # ResNet-34 architecture definition
│   ├── inference.py          # SageMaker inference handlers
│   └── requirements.txt      # Runtime dependencies
├── cloudformation.yaml       # AWS infrastructure (IaC)
├── deploy.sh                 # One-command deployment script
├── test_endpoint.py          # Python client for testing
└── delete_endpoint.py        # Cleanup script
```

## Prerequisites

- AWS CLI configured (`aws configure`)
- AWS account with SageMaker permissions
- Model weights file: `resnet34_flowers_best.pth` (place in parent directory)

## Quick Start

### 1. Deploy

```bash
./deploy.sh
```

This will:
- Package model weights + inference code into `model.tar.gz`
- Upload to S3
- Create SageMaker Model, Endpoint Config, and Endpoint via CloudFormation

Deployment takes ~5-7 minutes.

### 2. Test

```bash
python test_endpoint.py --image path/to/flower.jpg
```

Or using AWS CLI:

```bash
aws sagemaker-runtime invoke-endpoint \
    --endpoint-name <endpoint-name> \
    --content-type image/jpeg \
    --body fileb://flower.jpg \
    --region us-east-1 \
    output.json
```

### 3. Cleanup

```bash
aws cloudformation delete-stack --stack-name <stack-name> --region us-east-1
```

## API Reference

### Request

- **Endpoint**: `https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint-name}/invocations`
- **Method**: POST
- **Content-Type**: `image/jpeg` or `image/png`
- **Body**: Raw image bytes

### Response

```json
{
    "predictions": [
        {"class_id": 0, "class_name": "001", "probability": 0.9406},
        {"class_id": 12, "class_name": "013", "probability": 0.0044},
        ...
    ],
    "predicted_class": "001",
    "confidence": 0.9406
}
```

## Architecture

```
┌──────────────┐     ┌─────────────┐     ┌──────────────────────────────┐
│    Client    │────▶│   S3        │     │   SageMaker Endpoint         │
│  (HTTP POST) │     │ model.tar.gz│────▶│   - PyTorch 2.0 Container    │
└──────────────┘     └─────────────┘     │   - ml.t2.medium (CPU)       │
                                         │   - Auto-scaling ready       │
                                         └──────────────────────────────┘
```

### Inference Flow

```
Image bytes → input_fn() → predict_fn() → output_fn() → JSON response
              (preprocess)   (forward pass)  (format)
```

## Configuration

Edit `cloudformation.yaml` to customize:

| Parameter | Default | Options |
|-----------|---------|---------|
| InstanceType | ml.t2.medium | ml.t2.large, ml.m5.large, ml.g4dn.xlarge (GPU) |
| InitialInstanceCount | 1 | 1-10 |
| Region | us-east-1 | Any AWS region |

## Cost

| Instance | Cost (approx) |
|----------|---------------|
| ml.t2.medium | ~$0.05/hour |
| ml.m5.large | ~$0.12/hour |
| ml.g4dn.xlarge (GPU) | ~$0.74/hour |

**Remember to delete the endpoint when not in use to avoid charges.**

## Inference Code Details

The `inference.py` implements 4 SageMaker handler functions:

```python
model_fn(model_dir)      # Load model weights (called once at startup)
input_fn(body, type)     # Preprocess: image bytes → normalized tensor
predict_fn(tensor, model) # Forward pass through ResNet-34
output_fn(output, type)  # Format: logits → top-5 predictions JSON
```

## Dataset

Trained on the [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/):
- 8,189 images
- 102 flower categories
- 80/10/10 train/val/test split

## License

MIT
