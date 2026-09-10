# Reset Test Data And Enable MinIO

Use this procedure once, immediately before clean production acceptance testing.

## Existing reference locations

With `STORAGE_BACKEND=local`, current encrypted object-store references use:

```text
/opt/sental-proctor-service/storage/objects/proctoring-evidence/references/{companyId}/user_{userId}/
```

References created by the earlier implementation use:

```text
/opt/sental-proctor-service/storage/references/{companyId}/user_{userId}/template.json
/opt/sental-proctor-service/storage/references/{companyId}/user_{userId}/reference-*.jpg
```

The existing server inventory showed eight legacy files in `storage/references` and no files in
`storage/objects`.

## 1. Deploy the Moodle purge command

Deploy the latest `local_proctorcore` code, including `cli/purge_test_data.php`, to the Moodle server.

## 2. Enter maintenance mode and disable ProctorCore

```bash
cd /www/wwwroot/sental.kz
php admin/cli/maintenance.php --enable
php admin/cli/cfg.php --component=local_proctorcore --name=enabled --set=0
```

## 3. Back up ProctorCore test records

A new full Moodle backup is not required if the client already has a verified site backup. For this reset,
a small dump of the ProctorCore tables is sufficient because the purge command does not modify Moodle core,
quiz, IOMAD, company-policy, or quiz-configuration records.

Use the database host, user, and database name from `config.php`. Keep the password out of shell history by
using the interactive `-p` prompt. This command assumes the confirmed `mdl_` prefix used on this site.

```bash
set -o pipefail
mysqldump --single-transaction --quick --default-character-set=utf8mb4 \
  -h DB_HOST -u DB_USER -p DB_NAME \
  mdl_local_proctorcore_appeals \
  mdl_local_proctorcore_assets \
  mdl_local_proctorcore_audit \
  mdl_local_proctorcore_checks \
  mdl_local_proctorcore_companycfg \
  mdl_local_proctorcore_faceenrol \
  mdl_local_proctorcore_fields \
  mdl_local_proctorcore_fieldvals \
  mdl_local_proctorcore_quizcfg \
  mdl_local_proctorcore_rulesack \
  mdl_local_proctorcore_sessions \
  mdl_local_proctorcore_violations \
  mdl_local_proctorcore_webhooks \
  | gzip > /root/proctorcore-tables-before-reset.sql.gz
gzip -t /root/proctorcore-tables-before-reset.sql.gz
```

Generated test PDFs in Moodle's managed file area are intentionally disposable and are removed through the
Moodle file API. This table dump is therefore a safety snapshot of records, not a complete Moodle file backup.

Confirm that the existing Proctoring Server backup is readable:

```bash
gzip -t /root/sental-proctor-storage-before-minio.tar.gz
tar -tzf /root/sental-proctor-storage-before-minio.tar.gz | head -30
sha256sum /root/sental-proctor-storage-before-minio.tar.gz
```

## 4. Preview and purge Moodle ProctorCore test records

The first command is a dry run. Review every count before executing the second command.

```bash
cd /www/wwwroot/sental.kz
php local/proctorcore/cli/purge_test_data.php
php local/proctorcore/cli/purge_test_data.php --execute --confirm=DELETE-ALL-PROCTORCORE-TEST-DATA
php local/proctorcore/cli/purge_test_data.php
```

The final dry run must show zero rows for every `Delete` table. Company policies, quiz configuration, and
participant field definitions remain in place.

Delete test quiz attempts separately from the quiz's **Results > Grades** page. Select only the known test
attempts and use Moodle's **Delete selected attempts** action.

## 5. Stop Proctoring Server writers

```bash
cd /opt/sental-proctor-service
docker compose stop api worker livekit-egress livekit
docker compose ps
```

Do not continue if a real exam is active.

## 6. Quarantine local test storage and clear transient state

```bash
cd /opt/sental-proctor-service
STAMP=$(date +%Y%m%d-%H%M%S)
mv storage "storage.testdata-${STAMP}"
install -d -m 0750 storage
docker compose exec -T redis redis-cli FLUSHDB
```

`FLUSHDB` is acceptable here only because this Redis database is dedicated to this Compose stack, LiveKit is
stopped, and all existing sessions are test sessions.

Do not remove the MinIO volume. Its current contents are fresh MinIO metadata and it is the production object
store volume.

## 7. Enable production MinIO storage

Edit `/opt/sental-proctor-service/.env` and ensure these exact settings exist:

```dotenv
APP_ENV=production
STORAGE_BACKEND=minio
STORAGE_REQUIRE_READY=true
REFERENCE_ENCRYPTION_KEY_FILE=/run/secrets/reference_encryption_key
CORS_ORIGINS=https://sental.kz
S3_ENDPOINT=http://minio:9000
S3_SECURE=false
```

Do not change `S3_ACCESS_KEY`, `S3_SECRET_KEY`, bucket names, or the reference encryption key during this
cutover. Confirm the key is present without printing it:

```bash
test -s /opt/sental-proctor-service/secrets/reference_encryption_key && echo "Encryption key present"
```

Continue using the CPU-compatible MinIO override on this host:

```text
quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z-cpuv1
```

## 8. Start and verify the stack

```bash
cd /opt/sental-proctor-service
docker compose up -d --build --remove-orphans
docker compose ps
docker compose exec api python /app/scripts/verify_deployment.py --url http://127.0.0.1:8091
docker compose exec api python -c 'from app.core.config import get_settings; s=get_settings(); print({"environment": s.app_env, "storage": s.storage_backend})'
curl -fsS https://proctoring.sental.kz/api/health
```

The runtime check must report `production` and `minio`. Deployment verification and public health must pass.

Confirm the private buckets are reachable:

```bash
docker compose exec api python -c 'from app.services.object_store import ObjectStore; o=ObjectStore(); print([x["Name"] for x in o.client.list_buckets().get("Buckets", [])])'
```

Expected buckets:

```text
proctoring-temp
proctoring-evidence
proctoring-reports
```

## 9. Re-enable Moodle and run one acceptance attempt

```bash
cd /www/wwwroot/sental.kz
php admin/cli/cfg.php --component=local_proctorcore --name=enabled --set=1
php admin/cli/purge_caches.php
php admin/cli/maintenance.php --disable
```

Run one new enrollment and one complete proctored attempt. Confirm object counts afterward:

```bash
cd /opt/sental-proctor-service
docker compose exec api python -c 'from app.services.object_store import ObjectStore; o=ObjectStore(); print({b: sum(p.get("KeyCount", 0) for p in o.client.get_paginator("list_objects_v2").paginate(Bucket=b)) for b in o.buckets})'
```

The evidence bucket must contain the encrypted face template and accepted reference image after enrollment.
The completed attempt must produce its expected media evidence and Moodle report records.

## 10. Remove quarantined test files later

Keep `storage.testdata-*` only until the clean acceptance attempt and reports have been checked. The existing tar
archive remains the rollback copy. After approval, list the directory carefully and remove that specific stamped
directory; never use a wildcard and never run `docker compose down -v`.
