#!/usr/bin/env bash
# Argus Ultimate deployment script — Push 64
# Usage: ./deploy/scripts/deploy.sh [registry] [tag]
set -euo pipefail

REGISTRY="${1:-ghcr.io/toxicspawn}"
TAG="${2:-latest}"
IMAGE="${REGISTRY}/argus-ultimate:${TAG}"

echo "🚀 Building Argus Ultimate image: ${IMAGE}"
docker build \
  --file Dockerfile \
  --target runtime \
  --tag "${IMAGE}" \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
  .

echo "📤 Pushing image..."
docker push "${IMAGE}"

echo "⚙️  Applying Kubernetes manifests..."
kubectl apply -k deploy/k8s/

echo "🔄 Rolling out deployment..."
kubectl rollout restart deployment/argus -n argus-system
kubectl rollout status deployment/argus -n argus-system --timeout=120s

echo "✅ Deploy complete: ${IMAGE}"
