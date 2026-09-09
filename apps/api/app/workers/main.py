from redis import Redis
from rq import Queue, Worker

from app.core.config import get_settings
from app.services.object_store import ObjectStore


def main() -> None:
    settings = get_settings()
    ObjectStore(settings).ensure_ready()
    connection = Redis.from_url(settings.redis_url)
    worker = Worker([Queue(settings.queue_name, connection=connection)], connection=connection)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
