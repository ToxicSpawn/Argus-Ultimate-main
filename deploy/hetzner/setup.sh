#!/usr/bin/env bash
# deploy/hetzner/setup.sh
# Push 85 — Automated Hetzner k3s + Argus deployment
# Usage: bash deploy/hetzner/setup.sh <server-ip>

set -euo pipefail

SERVER_IP="${1:-}"
NAMESPACE="argus"
RELEASE="argus"
CHART_DIR="$(dirname "$0")/../../helm/argus"
VALUES_FILE="$(dirname "$0")/values-prod.yaml"

if [[ -z "$SERVER_IP" ]]; then
  echo "Usage: $0 <hetzner-server-ip>"
  exit 1
fi

echo "🚀 Argus Ultimate — Hetzner k3s Setup (Push 85)"
echo "Server: $SERVER_IP"
echo "Namespace: $NAMESPACE"
echo ""

# --- 1. Fetch kubeconfig
echo "[1/7] Fetching kubeconfig..."
scp root@"$SERVER_IP":/etc/rancher/k3s/k3s.yaml /tmp/argus-k3s.yaml
sed -i "s/127.0.0.1/$SERVER_IP/g" /tmp/argus-k3s.yaml
export KUBECONFIG=/tmp/argus-k3s.yaml
kubectl get nodes

# --- 2. Create namespace
echo "[2/7] Creating namespace $NAMESPACE..."
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# --- 3. Create secrets (from env)
echo "[3/7] Creating argus-secrets..."
if [[ -z "${ARGUS_API_KEY:-}" || -z "${ARGUS_API_SECRET:-}" ]]; then
  echo "WARNING: ARGUS_API_KEY / ARGUS_API_SECRET not set in environment."
  echo "         Create secrets manually before trading live."
else
  kubectl create secret generic argus-secrets \
    --namespace "$NAMESPACE" \
    --from-literal=ARGUS_API_KEY="${ARGUS_API_KEY}" \
    --from-literal=ARGUS_API_SECRET="${ARGUS_API_SECRET}" \
    --from-literal=ARGUS_TELEGRAM_BOT_TOKEN="${ARGUS_TELEGRAM_BOT_TOKEN:-}" \
    --from-literal=ARGUS_TELEGRAM_CHAT_ID="${ARGUS_TELEGRAM_CHAT_ID:-}" \
    --dry-run=client -o yaml | kubectl apply -f -
  echo "   Secrets applied."
fi

# --- 4. Helm repos
echo "[4/7] Adding Helm repos..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo add grafana https://grafana.github.io/helm-charts 2>/dev/null || true
helm repo update

# --- 5. Helm deps
echo "[5/7] Updating chart dependencies..."
helm dependency update "$CHART_DIR"

# --- 6. Deploy
echo "[6/7] Deploying Argus..."
helm upgrade --install "$RELEASE" "$CHART_DIR" \
  --namespace "$NAMESPACE" \
  -f "$VALUES_FILE" \
  --atomic \
  --timeout 10m \
  --wait

# --- 7. Status
echo "[7/7] Deployment status:"
kubectl get pods -n "$NAMESPACE"
kubectl get svc -n "$NAMESPACE"
kubectl get pvc -n "$NAMESPACE"

echo ""
echo "✅ Argus deployed successfully!"
echo ""
echo "  Access Grafana:"
echo "    kubectl port-forward -n $NAMESPACE svc/$RELEASE-grafana 3000:3000"
echo "    http://localhost:3000 (admin / argus-admin)"
echo ""
echo "  Stream logs:"
echo "    kubectl logs -n $NAMESPACE -l app.kubernetes.io/name=argus -f"
