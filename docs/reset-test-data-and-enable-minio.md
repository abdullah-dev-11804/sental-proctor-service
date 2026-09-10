# Clean Test Data And Enable MinIO

Use this runbook once for the production storage cutover. It deliberately removes disposable ProctorCore test
records while preserving Moodle quiz configuration, company policy, participant field definitions, models,
secrets, and the MinIO Docker volume.

Commands are labelled by server. Do not paste a command into the other server.

## What this procedure preserves

- Moodle users, courses, quizzes, IOMAD companies, question attempts, and grades.
- `local_proctorcore` company settings, quiz settings, and participant field definitions.
- `/opt/sental-proctor-service/models`.
- `/opt/sental-proctor-service/secrets/reference_encryption_key`.
- `.env`, Compose configuration, and the CPU-compatible MinIO override.
- The existing MinIO Docker volume.
- The existing safety archive `/root/sental-proctor-storage-before-minio.tar.gz`.

## What this procedure resets

- ProctorCore sessions, violations, assets, appeals, checks, acknowledgements, audit entries, webhooks, and
  face-enrollment records.
- ProctorCore-generated report files in Moodle's managed file area.
- Proctoring Server local session manifests, local media, legacy references, and debug artifacts.
- The dedicated Redis database used by this Compose stack.

Moodle quiz attempts are not automatically deleted. Delete only known test attempts from the quiz's
**Results > Grades** page if a completely clean quiz history is required.

## Existing reference locations

Before the cutover, encrypted references written through the local object-store backend are under:

```text
/opt/sental-proctor-service/storage/objects/proctoring-evidence/references/{companyId}/user_{userId}/
```

Older implementation files are under:

```text
/opt/sental-proctor-service/storage/references/{companyId}/user_{userId}/template.json
/opt/sental-proctor-service/storage/references/{companyId}/user_{userId}/reference-*.jpg
```

After the cutover, new encrypted templates and reference images are stored in the private
`proctoring-evidence` MinIO bucket under `references/{companyId}/user_{userId}/`.

## 1. Preflight checks

### On the Moodle server

Confirm that the purge command is deployed and preview the records:

```bash
cd /www/wwwroot/sental.kz
test -f local/proctorcore/cli/purge_test_data.php && echo "Purge command present"
php local/proctorcore/cli/purge_test_data.php
```

The second command is a dry run. It changes nothing. Also confirm that no real exam is active.

### On the Proctoring Server

```bash
cd /opt/sental-proctor-service
docker compose ps
docker compose exec api python /app/scripts/verify_deployment.py \
  --url http://127.0.0.1:8091
test -s secrets/reference_encryption_key && echo "Encryption key present"
gzip -t /root/sental-proctor-storage-before-minio.tar.gz
tar -tzf /root/sental-proctor-storage-before-minio.tar.gz | head -30
sha256sum /root/sental-proctor-storage-before-minio.tar.gz
```

Stop here if an actual exam is active, verification fails, the encryption key is absent, or the archive is
unreadable.

Confirm the MinIO image selected by Compose:

```bash
docker compose config --images
```

This older-CPU host must select:

```text
quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z-cpuv1
```

The server's existing `docker-compose.override.yml` must therefore contain:

```yaml
services:
  minio:
    image: quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z-cpuv1
```

Do not remove this override during deployment or Git cleanup.

## 2. Block new sessions and stop all writers

### On the Moodle server

```bash
cd /www/wwwroot/sental.kz
php admin/cli/maintenance.php --enable
php admin/cli/cfg.php --component=local_proctorcore --name=enabled --set=0
php admin/cli/purge_caches.php
```

### On the Proctoring Server

```bash
cd /opt/sental-proctor-service
docker compose stop api worker livekit-egress livekit
docker compose ps
```

Redis and MinIO may remain running. Do not continue if API, worker, Egress, or LiveKit still reports `Up`.

## 3. Back up the Moodle database

### On the Moodle server

Unless the client has explicitly confirmed a recent, restorable database backup, take a complete Moodle
database dump. Do not assume that a hosting snapshot or an unspecified backup exists.

Display the connection identifiers from `config.php` without printing the database password:

```bash
cd /www/wwwroot/sental.kz
php -r 'define("CLI_SCRIPT", true); require "config.php"; printf("host=%s\ndatabase=%s\nuser=%s\nprefix=%s\n", $CFG->dbhost, $CFG->dbname, $CFG->dbuser, $CFG->prefix);'
```

Use those values below. Check the estimated logical database size and free disk space first:

```bash
mysql -h DB_HOST -u DB_USER -p -N -e \
  "SELECT CONCAT(ROUND(SUM(data_length + index_length) / 1024 / 1024, 1), ' MB') FROM information_schema.tables WHERE table_schema = 'DB_NAME';"
df -h /root
```

For a conservative margin, `/root` should have free space at least equal to the reported logical database size,
even though the streamed gzip file will normally be smaller. Create and verify the complete dump:

```bash
cd /www/wwwroot/sental.kz
(
  umask 077
  set -o pipefail
  mysqldump --single-transaction --quick --skip-lock-tables --hex-blob \
    --default-character-set=utf8mb4 \
    -h DB_HOST -u DB_USER -p DB_NAME \
    | gzip -1 > /root/moodle-full-before-proctor-minio.sql.gz
)
gzip -t /root/moodle-full-before-proctor-minio.sql.gz
ls -lh /root/moodle-full-before-proctor-minio.sql.gz
sha256sum /root/moodle-full-before-proctor-minio.sql.gz \
  > /root/moodle-full-before-proctor-minio.sql.gz.sha256
cat /root/moodle-full-before-proctor-minio.sql.gz.sha256
```

Every command must succeed. A non-empty file and a successful `gzip -t` are mandatory before continuing.
Keep this dump until the MinIO cutover and production observation period are accepted.

Also create a smaller, data-only dump of exactly the ProctorCore tables the purge command deletes. This is
convenient for investigation, but it does not replace the full database dump. The commands below use the
confirmed `mdl_` prefix; change every prefix if `config.php` reports something else.

```bash
cd /www/wwwroot/sental.kz
(
  umask 077
  set -o pipefail
  mysqldump --single-transaction --quick --default-character-set=utf8mb4 \
    --no-create-info --skip-triggers --complete-insert \
    -h DB_HOST -u DB_USER -p DB_NAME \
    mdl_local_proctorcore_appeals \
    mdl_local_proctorcore_assets \
    mdl_local_proctorcore_audit \
    mdl_local_proctorcore_checks \
    mdl_local_proctorcore_faceenrol \
    mdl_local_proctorcore_fieldvals \
    mdl_local_proctorcore_rulesack \
    mdl_local_proctorcore_sessions \
    mdl_local_proctorcore_violations \
    mdl_local_proctorcore_webhooks \
    | gzip > /root/proctorcore-transactional-before-minio.sql.gz
)
gzip -t /root/proctorcore-transactional-before-minio.sql.gz
ls -lh /root/proctorcore-transactional-before-minio.sql.gz
```

The password is requested interactively and is not placed in shell history. This dump does not include the
generated report files removed through Moodle's file API; those reports are disposable test evidence.

## 4. Purge Moodle ProctorCore test records

### On the Moodle server

Run another dry run, review every count, execute the purge, and confirm that the deletion set is empty:

```bash
cd /www/wwwroot/sental.kz
php local/proctorcore/cli/purge_test_data.php
php local/proctorcore/cli/purge_test_data.php \
  --execute --confirm=DELETE-ALL-PROCTORCORE-TEST-DATA
php local/proctorcore/cli/purge_test_data.php
```

The final output must show zero rows for every `Delete` table. The `Preserve` rows may remain non-zero.

## 5. Quarantine local storage and clear transient state

### On the Proctoring Server

The old directory is moved, not immediately deleted. Its exact path is recorded for validation or rollback.

```bash
cd /opt/sental-proctor-service
STAMP=$(date +%Y%m%d-%H%M%S)
QUARANTINE="/opt/sental-proctor-service/storage.testdata-${STAMP}"
test -d storage
mv storage "$QUARANTINE"
install -d -m 0750 storage
printf '%s\n' "$QUARANTINE" > /root/sental-proctor-quarantine-path
cat /root/sental-proctor-quarantine-path
du -sh "$QUARANTINE"
docker compose exec -T redis redis-cli FLUSHDB
docker compose exec -T redis redis-cli DBSIZE
```

`DBSIZE` must return `0`. `FLUSHDB` is permitted only because Redis database `0` is dedicated to this stack,
all prior sessions are test sessions, and its writers are stopped.

Do not delete any of these:

```text
/opt/sental-proctor-service/models
/opt/sental-proctor-service/secrets
/opt/sental-proctor-service/.env
/var/lib/docker/volumes/sental-proctor-service_minio_data
/root/sental-proctor-storage-before-minio.tar.gz
```

Never run `docker compose down -v`; `-v` would remove persistent Docker volumes.

## 6. Enable MinIO in the application

### On the Proctoring Server

Open the environment file:

```bash
cd /opt/sental-proctor-service
vi .env
```

Ensure there is only one active definition of each setting and that these values are present:

```dotenv
APP_ENV=production
STORAGE_BACKEND=minio
STORAGE_REQUIRE_READY=true
REFERENCE_ENCRYPTION_KEY_FILE=/run/secrets/reference_encryption_key
CORS_ORIGINS=https://sental.kz
S3_ENDPOINT=http://minio:9000
S3_SECURE=false
IDENTITY_PASS_THRESHOLD=0.85
IDENTITY_REVIEW_THRESHOLD=0.70
IDENTITY_REQUIRE_PASSIVE_ANTISPOOF=true
IDENTITY_ANTISPOOF_MODEL=minifasnet_v2.onnx
IDENTITY_HEADPOSE_MODEL=SixDRepNet.onnx
IDENTITY_REQUIRE_HEADPOSE_LIVENESS=false
LIVEKIT_EGRESS_ENABLED=true
LIVEKIT_EGRESS_HEALTH_URL=http://livekit-egress:9090/metrics
MOODLE_WEBHOOK_URL=https://sental.kz/local/proctorcore/webhook.php
```

Keep the existing `S3_ACCESS_KEY`, `S3_SECRET_KEY`, bucket names, API secret, LiveKit secret, Moodle webhook
secret, and encryption key unchanged. `IDENTITY_REQUIRE_HEADPOSE_LIVENESS=false` prevents an admission-time head
turn challenge; SixDRepNet is still loaded and used for sustained look-away monitoring. Check non-secret resolved
values:

```bash
grep -E '^(APP_ENV|STORAGE_BACKEND|STORAGE_REQUIRE_READY|REFERENCE_ENCRYPTION_KEY_FILE|CORS_ORIGINS|S3_ENDPOINT|S3_SECURE|IDENTITY_PASS_THRESHOLD|IDENTITY_REVIEW_THRESHOLD|IDENTITY_REQUIRE_PASSIVE_ANTISPOOF|IDENTITY_ANTISPOOF_MODEL|IDENTITY_HEADPOSE_MODEL|IDENTITY_REQUIRE_HEADPOSE_LIVENESS|LIVEKIT_EGRESS_ENABLED|LIVEKIT_EGRESS_HEALTH_URL|MOODLE_WEBHOOK_URL)=' .env
test -s secrets/reference_encryption_key && echo "Encryption key present"
docker compose config --quiet
docker compose config --images
```

## 7. Start and verify the complete stack

### On the Proctoring Server

```bash
cd /opt/sental-proctor-service
docker compose up -d --build --remove-orphans
docker compose ps
docker compose exec api python /app/scripts/verify_deployment.py \
  --url http://127.0.0.1:8091
docker compose exec api python -c 'from app.core.config import get_settings; s=get_settings(); print({"environment": s.app_env, "storage": s.storage_backend})'
curl -fsS https://proctoring.sental.kz/api/health
```

The runtime command must print `production` and `minio`. All six services must be running and deployment
verification must report `"ok": true`.

Confirm that all private buckets exist and start empty except for MinIO metadata:

```bash
docker compose exec api python -c 'from app.services.object_store import ObjectStore; o=ObjectStore(); print(o.status())'
docker compose exec api python -c 'from app.services.object_store import ObjectStore; o=ObjectStore(); print({b: sum(p.get("KeyCount", 0) for p in o.client.get_paginator("list_objects_v2").paginate(Bucket=b)) for b in o.buckets})'
```

Expected bucket names are `proctoring-temp`, `proctoring-evidence`, and `proctoring-reports`. Do not continue
to Moodle acceptance testing unless storage reports `ready: True` and `private: True`.

## 8. Re-enable Moodle

### On the Moodle server

```bash
cd /www/wwwroot/sental.kz
php admin/cli/cfg.php --component=local_proctorcore --name=enabled --set=1
php admin/cli/purge_caches.php
php admin/cli/maintenance.php --disable
```

## 9. Run clean acceptance tests

1. Start the first attempt for a test user with no reference and complete enrollment.
2. Complete and submit that proctored attempt.
3. Start a second attempt for the same user and confirm stored-reference verification occurs.
4. During the second attempt, trigger a controlled tab switch, sustained look-away, and temporary no-face event.
5. Submit the second attempt and wait for evidence processing to finish.
6. In Moodle, confirm the session report contains the identity result, timestamped violations, snapshots, clips,
   recording state, and retention dates.

### On the Proctoring Server

Check logs and MinIO object counts after the attempts:

```bash
cd /opt/sental-proctor-service
docker compose logs --since=30m api worker livekit-egress
docker compose exec api python -c 'from app.services.object_store import ObjectStore; o=ObjectStore(); print({b: sum(p.get("KeyCount", 0) for p in o.client.get_paginator("list_objects_v2").paginate(Bucket=b)) for b in o.buckets})'
docker compose exec api python -c 'from app.services.object_store import ObjectStore; o=ObjectStore(); print({b: [x["Key"] for p in o.client.get_paginator("list_objects_v2").paginate(Bucket=b) for x in p.get("Contents", [])][:20] for b in o.buckets})'
```

The `proctoring-evidence` bucket must contain encrypted reference objects and company/session-scoped evidence.
Expected session assets must also appear in Moodle. Do not approve the cutover if objects exist only under the
new local `storage` directory.

## 10. Roll back if acceptance fails

Rollback returns the application to local storage. It does not restore disposable Moodle report files.

### On the Moodle server

```bash
cd /www/wwwroot/sental.kz
php admin/cli/maintenance.php --enable
php admin/cli/cfg.php --component=local_proctorcore --name=enabled --set=0
php admin/cli/purge_caches.php
```

### On the Proctoring Server

```bash
cd /opt/sental-proctor-service
docker compose stop api worker livekit-egress livekit
QUARANTINE=$(cat /root/sental-proctor-quarantine-path)
case "$QUARANTINE" in
  /opt/sental-proctor-service/storage.testdata-*) ;;
  *) echo "Unsafe quarantine path: $QUARANTINE"; exit 1 ;;
esac
test -d "$QUARANTINE"
FAILED="/opt/sental-proctor-service/storage.failed-minio-$(date +%Y%m%d-%H%M%S)"
mv storage "$FAILED"
mv "$QUARANTINE" storage
vi .env
```

Set `STORAGE_BACKEND=local`, retain the encryption-key settings, then restart:

```bash
docker compose up -d --build --remove-orphans
docker compose exec api python /app/scripts/verify_deployment.py \
  --url http://127.0.0.1:8091
```

Because production readiness intentionally requires MinIO, a production-mode local-storage rollback may report
not ready. Keep Moodle disabled while diagnosing, or restore the previously backed-up `.env` and Compose files
from the archive only after inspecting them.

The transactional SQL dump is an emergency copy of old test records. Normally do not restore it: those records
refer to disposable report files that were intentionally removed. If restoration is genuinely required, do it
only in maintenance mode with a database administrator.

## 11. Permanently delete quarantined test files after approval

Do this only after both acceptance attempts, Moodle reports, MinIO objects, reconnect behavior, and retention
metadata have been checked and accepted. Keep the tar archive through the production observation period.

### On the Proctoring Server

```bash
QUARANTINE=$(cat /root/sental-proctor-quarantine-path)
case "$QUARANTINE" in
  /opt/sental-proctor-service/storage.testdata-*) ;;
  *) echo "Unsafe quarantine path: $QUARANTINE"; exit 1 ;;
esac
test -d "$QUARANTINE"
du -sh "$QUARANTINE"
find "$QUARANTINE" -maxdepth 1 -mindepth 1 -printf '%f\n' | head -50
```

After confirming the displayed path is the quarantined test-data directory, remove that exact path:

```bash
rm -rf --one-file-system "$QUARANTINE"
test ! -e "$QUARANTINE" && echo "Quarantined test data removed"
rm -f /root/sental-proctor-quarantine-path
```

Do not remove `/root/sental-proctor-storage-before-minio.tar.gz` yet. A single MinIO Docker volume is durable
primary storage, not an independent backup; off-host backup or replication remains a production operational
risk to resolve separately.
