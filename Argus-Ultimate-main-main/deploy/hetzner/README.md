# Argus Ultimate — Hetzner k3s Deployment (Push 85)

> Single-node k3s cluster on Hetzner CX32/CX52, Helm-managed.

## Prerequisites

- Hetzner account with a CX32 or CX52 instance (Ubuntu 22.04)
- `kubectl`, `helm` v3.14+ installed locally
- Docker image built and pushed to `ghcr.io/toxicspawn/argus-ultimate`
- Domain (optional, for Traefik ingress + TLS)

---

## 1. Provision Hetzner Server

```bash
# Via hcloud CLI
hcloud server create \
  --name argus-k3s \
  --type cx32 \
  --image ubuntu-22.04 \
  --location fsn1 \
  --ssh-key your-key
```

---

## 2. Install k3s

```bash
ssh root@<server-ip>

# Install k3s (single-node, embedded etcd disabled for single node)
curl -sfL https://get.k3s.io | sh -s - \
  --disable traefik \
  --write-kubeconfig-mode 644

# Wait for node ready
kubectl get nodes

# Copy kubeconfig locally
scp root@<server-ip>:/etc/rancher/k3s/k3s.yaml ~/.kube/argus-k3s.yaml
sed -i 's/127.0.0.1/<server-ip>/g' ~/.kube/argus-k3s.yaml
export KUBECONFIG=~/.kube/argus-k3s.yaml
```

---

## 3. Install Traefik (optional, for ingress)

```bash
helm repo add traefik https://traefik.github.io/charts
helm repo update
helm install traefik traefik/traefik \
  --namespace kube-system \
  --set ports.websecure.tls.enabled=true
```

---

## 4. Create namespace + secrets

```bash
kubectl create namespace argus

kubectl create secret generic argus-secrets \
  --namespace argus \
  --from-literal=ARGUS_API_KEY=your_binance_key \
  --from-literal=ARGUS_API_SECRET=your_binance_secret \
  --from-literal=ARGUS_TELEGRAM_BOT_TOKEN=your_bot_token \
  --from-literal=ARGUS_TELEGRAM_CHAT_ID=your_chat_id
```

---

## 5. Add Helm repos

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Download chart dependencies
cd helm/argus
helm dependency update
```

---

## 6. Deploy

```bash
# Dry run first
helm install argus ./helm/argus \
  --namespace argus \
  --dry-run --debug

# Deploy
helm install argus ./helm/argus \
  --namespace argus \
  --set argus.image.tag=8.21.0

# Watch rollout
kubectl rollout status deployment/argus -n argus
kubectl get pods -n argus
```

---

## 7. Access dashboards

```bash
# Grafana
kubectl port-forward -n argus svc/argus-grafana 3000:3000 &
open http://localhost:3000
# Login: admin / argus-admin

# Prometheus
kubectl port-forward -n argus svc/argus-prometheus-server 9090:80 &
open http://localhost:9090

# Argus metrics
kubectl port-forward -n argus svc/argus 8080:8080 &
curl http://localhost:8080/metrics
```

---

## 8. Upgrade

```bash
# Build + push new image
docker build -t ghcr.io/toxicspawn/argus-ultimate:8.22.0 .
docker push ghcr.io/toxicspawn/argus-ultimate:8.22.0

# Upgrade Helm release
helm upgrade argus ./helm/argus \
  --namespace argus \
  --set argus.image.tag=8.22.0
```

---

## 9. Rollback

```bash
helm history argus -n argus
helm rollback argus <revision> -n argus
```

---

## 10. Uninstall

```bash
helm uninstall argus -n argus
kubectl delete namespace argus
```

---

## Resource Recommendations

| Server | vCPU | RAM | Use case |
|---|---|---|---|
| CX22 | 2 | 4GB | Paper trading only |
| CX32 | 4 | 8GB | Live trading (recommended) |
| CX52 | 8 | 16GB | Live + Grafana + full stack |
| CCX33 | 8 | 32GB | HFT + ML inference |

---

## Hetzner Firewall Rules

```bash
hcloud firewall create --name argus-fw
hcloud firewall add-rule argus-fw --direction in --protocol tcp --port 22 --source-ips 0.0.0.0/0
hcloud firewall add-rule argus-fw --direction in --protocol tcp --port 6443 --source-ips <your-ip>/32
hcloud firewall apply-to-server argus-fw --server argus-k3s
```

Only expose port 6443 (k3s API) to your IP. All other access via `kubectl port-forward`.

---

## Production values override

Create `deploy/hetzner/values-prod.yaml`:

```yaml
argus:
  image:
    tag: "8.21.0"
  resources:
    requests:
      cpu: "1000m"
      memory: "1Gi"
    limits:
      cpu: "3000m"
      memory: "4Gi"
  command: ["python", "run_ultimate.py"]
  env:
    - name: ARGUS_ENV
      value: production
    - name: TZ
      value: Australia/Sydney

grafana:
  adminPassword: "change-me-in-production"

prometheus:
  server:
    retention: "30d"
    persistentVolume:
      size: 50Gi
```

Then deploy with:
```bash
helm upgrade --install argus ./helm/argus \
  --namespace argus \
  -f deploy/hetzner/values-prod.yaml
```
