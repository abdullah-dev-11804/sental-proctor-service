from __future__ import annotations

from typing import Any

from app.core.config import Settings, get_settings


class JobQueue:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def enqueue(
        self,
        function: str,
        *args: Any,
        job_id: str | None = None,
        retry: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        from redis import Redis
        from rq import Queue, Retry
        from rq.exceptions import InvalidJobOperation
        from rq.job import Job

        connection = Redis.from_url(self.settings.redis_url)
        queue = Queue(self.settings.queue_name, connection=connection, default_timeout=14400)
        try:
            job = queue.enqueue(
                function,
                *args,
                kwargs=kwargs,
                job_id=job_id,
                result_ttl=86400,
                failure_ttl=604800,
                retry=Retry(
                    max=max(1, int(self.settings.webhook_max_attempts)),
                    interval=self.settings.webhook_retry_schedule,
                ) if retry else None,
            )
        except (InvalidJobOperation, ValueError):
            if not job_id:
                raise
            # Terminal media jobs are deliberately idempotent. If Moodle sends
            # the same stop twice, return the durable job already in Redis.
            job = Job.fetch(job_id, connection=connection)
        return {"jobId": job.id, "queue": queue.name, "status": job.get_status(refresh=False)}

    def status(self) -> dict[str, Any]:
        try:
            from redis import Redis
            from rq import Queue, Worker
            from rq.registry import DeferredJobRegistry, FailedJobRegistry, ScheduledJobRegistry, StartedJobRegistry

            connection = Redis.from_url(self.settings.redis_url, socket_timeout=2)
            queue = Queue(self.settings.queue_name, connection=connection)
            workers = len(Worker.all(connection=connection, queue=queue))
            return {
                "ready": bool(connection.ping()),
                "queue": queue.name,
                "queued": len(queue),
                "running": StartedJobRegistry(queue=queue).count,
                "scheduledRetries": ScheduledJobRegistry(queue=queue).count,
                "deferred": DeferredJobRegistry(queue=queue).count,
                "deadLetter": FailedJobRegistry(queue=queue).count,
                "workers": workers,
            }
        except Exception as exc:
            return {"ready": False, "queue": self.settings.queue_name, "error": str(exc)[:300]}
