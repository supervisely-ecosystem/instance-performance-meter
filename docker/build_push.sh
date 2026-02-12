#!/bin/bash

set -e

# Check if version is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 1.0.0"
    exit 1
fi

VERSION="$1"
IMAGE_NAME="supervisely/instance-performance"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Version: $VERSION"
echo "Building image: $IMAGE_NAME:$VERSION"

# Run download_sample.sh
echo "Running download_sample.sh..."
bash "$SCRIPT_DIR/download_sample.sh"

# Build Docker image
echo "Building Docker image..."
docker build -t "$IMAGE_NAME:$VERSION" -t "$IMAGE_NAME:latest" "$SCRIPT_DIR"

echo "Done! Image built: $IMAGE_NAME:$VERSION"
echo "To push the image, run:"
echo "  docker push $IMAGE_NAME:$VERSION"
echo "  docker push $IMAGE_NAME:latest"
