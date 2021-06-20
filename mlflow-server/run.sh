#!/bin/sh

mlflow server --host 0.0.0.0 --backend-store-uri ${MLFLOW__BACKEND_STORE_URI} --default-artifact-root ${DEFAULT_S3}