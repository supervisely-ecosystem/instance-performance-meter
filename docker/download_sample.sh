#!/bin/bash

set -e

IMAGE="supervisely/instance-performance:latest"
FILE_PATH="/tmp/Performance_Test.tar"
OUTPUT_FILE="Performance_Test.tar"

echo "Pulling Docker image: $IMAGE"
docker pull "$IMAGE"

echo "Creating temporary container..."
CONTAINER_ID=$(docker create "$IMAGE")

echo "Copying file from container..."
docker cp "$CONTAINER_ID:$FILE_PATH" "$OUTPUT_FILE"

echo "Removing temporary container..."
docker rm "$CONTAINER_ID"

echo "Done! File saved as: $OUTPUT_FILE"
