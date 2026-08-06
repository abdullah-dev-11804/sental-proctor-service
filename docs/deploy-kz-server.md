# Kazakh Server Deployment

Target server:

```text
Domain: vm8400.fst.kz
IP: 109.248.247.21
OS: Ubuntu 22.04
CPU/RAM/Disk: 4 vCPU, 16 GB RAM, 200 GB
```

This host is enough for staging, identity verification, and light testing. For
many simultaneous recorded exams, add monitoring and revisit CPU/disk capacity.

## Server Install

SSH into the server:

```bash
ssh root@109.248.247.21
```

Install Docker and Git:

```bash
apt update
apt install -y git ca-certificates curl gnupg ufw docker.io docker-compose-plugin
systemctl enable --now docker
docker compose version || apt install -y docker-compose-v2
docker compose version || apt install -y docker-compose
```

Open staging ports:

```bash
ufw allow OpenSSH
ufw allow 8091/tcp
ufw allow 9000/tcp
ufw allow 9001/tcp
ufw allow 7880/tcp
ufw allow 7881/tcp
ufw allow 50000:50100/udp
ufw --force enable
```

Clone the service:

```bash
cd /opt
git clone YOUR_GITHUB_REPO_URL sental-proctor-service
cd /opt/sental-proctor-service
```

Create the environment file:

```bash
cp .env.kazakhstan.example .env
```

Generate secrets:

```bash
openssl rand -base64 32
openssl rand -base64 32
openssl rand -base64 32
```

Put those values into:

```text
API_SHARED_SECRET
S3_SECRET_KEY
MOODLE_WEBHOOK_SECRET
```

Edit the file:

```bash
nano .env
```

Start Server B:

```bash
docker compose up -d --build
docker compose ps
```

Check health:

```bash
curl http://109.248.247.21:8091/api/health
curl http://vm8400.fst.kz:8091/api/health
```

Expected:

```json
{"ok":true,"status":"healthy"}
```

## Moodle Settings

In `local_proctorcore` settings:

```text
Enabled: Yes
Server B URL: http://vm8400.fst.kz:8091
Server API key: same as API_SHARED_SECRET
Webhook secret: same as MOODLE_WEBHOOK_SECRET
Verify SSL: No, while using plain HTTP
LiveKit browser client URL: empty
Identity threshold: 0.35 for the current baseline matcher
```

Also ensure Moodle HTTP security allows port `8091`.

## Redeploy

After pushing changes:

```bash
cd /opt/sental-proctor-service
git pull
docker compose up -d --build
docker compose logs -f api
```
