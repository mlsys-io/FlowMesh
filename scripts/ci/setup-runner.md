# FlowMesh CI — Self-Hosted Runner Setup

This guide sets up GitHub Actions self-hosted runners on the FlowMesh GPU and CPU machines.

## Overview

| Machine | Role | Labels |
|---------|------|--------|
| luyao3 | Integration tests (CPU) | `self-hosted,linux,luyao3` |
| luyao3 | GPU smoke tests | `self-hosted,linux,luyao3` |

Each machine runs one runner. Multiple runners on the same machine would cause GPU memory conflicts.

---

## Part 1 — Prerequisites (all machines)

### 1.1 Create a dedicated runner user

Run as root:

```bash
sudo useradd -m -s /bin/bash github-runner
sudo usermod -aG docker github-runner   # allow Docker without sudo
```

### 1.2 Install Docker

```bash
curl -fsSL https://get.docker.com | sudo bash
sudo systemctl enable --now docker
```

Verify:

```bash
docker run --rm hello-world
```

---

## Part 2 — GPU machines only (RTX 5080)

### 2.1 Install nvidia-container-toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify:

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

---

## Part 3 — Install the GitHub Actions runner

Repeat this section on **each machine** with the appropriate labels.

### 3.1 Get a runner registration token

In the GitHub repo:
**Settings → Actions → Runners → New self-hosted runner**

Copy the token shown (valid for 1 hour).

### 3.2 Download and configure the runner

Run as `github-runner` user:

```bash
sudo -u github-runner -i   # switch to runner user

mkdir -p ~/actions-runner && cd ~/actions-runner

# Download latest runner (check https://github.com/actions/runner/releases for latest version)
curl -sL https://github.com/actions/runner/releases/download/v2.322.0/actions-runner-linux-x64-2.322.0.tar.gz \
  -o actions-runner.tar.gz
tar xzf actions-runner.tar.gz
rm actions-runner.tar.gz
```

Configure — **luyao3 (CPU + GPU)**:

```bash
./config.sh \
  --url https://github.com/mlsys-io/FlowMesh \
  --token <TOKEN_FROM_GITHUB> \
  --name "luyao3" \
  --labels "self-hosted,linux,luyao3" \
  --work "_work" \
  --unattended
```

### 3.3 Install as a systemd service

```bash
# Still as github-runner user inside ~/actions-runner
exit   # back to root or sudo user

sudo /home/github-runner/actions-runner/svc.sh install github-runner
sudo /home/github-runner/actions-runner/svc.sh start
```

Verify the service is running:

```bash
sudo /home/github-runner/actions-runner/svc.sh status
# or
sudo systemctl status actions.runner.mlsys-io-FlowMesh.*.service
```

---

## Part 4 — GitHub Secrets

Add these in **Settings → Secrets and variables → Actions**:

| Secret | Value | Used by |
|--------|-------|---------|
| `HF_TOKEN` | HuggingFace API token | GPU worker (model downloads) |

The CI API key (`flm-ci-00000000000000000000000000000000`) is hardcoded in the CI compose and test script — it is a fixed test credential, not a real secret.

---

## Part 5 — Verify the runner appears in GitHub

Go to **Settings → Actions → Runners** in the repo.
Each machine should show as **Idle** within a minute of starting the service.

---

## Maintenance

### View runner logs

```bash
journalctl -u "actions.runner.*" -f
```

### Remove a runner

```bash
cd ~/actions-runner
sudo ./svc.sh stop
sudo ./svc.sh uninstall
./config.sh remove --token <TOKEN>
```

### Disk cleanup (CI build cache accumulates over time)

Add a cron job on each runner machine:

```bash
# As root — weekly Docker prune
echo "0 3 * * 0 root docker system prune -f --filter until=168h" \
  > /etc/cron.d/docker-prune
```
