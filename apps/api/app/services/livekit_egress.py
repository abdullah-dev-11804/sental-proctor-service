from __future__ import annotations

import asyncio
from typing import Any

from app.core.config import Settings, get_settings


class LiveKitEgress:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def start_participant(self, room: str, identity: str, object_prefix: str) -> dict[str, Any]:
        if not self.settings.livekit_egress_enabled:
            return {"enabled": False, "state": "browser_fallback"}
        return asyncio.run(self._start_participant(room, identity, object_prefix))

    async def _start_participant(self, room: str, identity: str, object_prefix: str) -> dict[str, Any]:
        from livekit import api

        upload = api.S3Upload(
            access_key=self.settings.s3_access_key,
            secret=self.settings.s3_secret_key,
            region=self.settings.s3_region,
            endpoint=self.settings.s3_endpoint,
            bucket=self.settings.s3_bucket_temp,
            force_path_style=True,
        )
        output = api.SegmentedFileOutput(
            protocol=api.HLS_PROTOCOL,
            filename_prefix=f"{object_prefix}/segment",
            playlist_name=f"{object_prefix}/index.m3u8",
            live_playlist_name=f"{object_prefix}/live.m3u8",
            segment_duration=max(2, int(self.settings.livekit_egress_segment_seconds)),
            s3=upload,
        )
        request = api.ParticipantEgressRequest(
            room_name=room,
            identity=identity,
            segment_outputs=[output],
        )
        client = api.LiveKitAPI(
            self.settings.livekit_internal_url,
            self.settings.livekit_api_key,
            self.settings.livekit_api_secret,
        )
        try:
            info = await client.egress.start_participant_egress(request)
            return {
                "enabled": True,
                "state": "active",
                "egressId": info.egress_id,
                "roomId": info.room_id,
                "objectPrefix": object_prefix,
                "startedAtNs": int(info.started_at),
            }
        finally:
            await client.aclose()

    def stop(self, egress_id: str) -> dict[str, Any]:
        if not egress_id:
            return {"state": "not_started"}
        return asyncio.run(self._stop(egress_id))

    async def _stop(self, egress_id: str) -> dict[str, Any]:
        from google.protobuf.json_format import MessageToDict
        from livekit import api

        client = api.LiveKitAPI(
            self.settings.livekit_internal_url,
            self.settings.livekit_api_key,
            self.settings.livekit_api_secret,
        )
        try:
            info = await client.egress.stop_egress(api.StopEgressRequest(egress_id=egress_id))
            result = MessageToDict(info, preserving_proto_field_name=False)
            result["state"] = "stopping"
            return result
        finally:
            await client.aclose()

    def status(self) -> dict[str, Any]:
        result = {
            "enabled": bool(self.settings.livekit_egress_enabled),
            "internalUrl": self.settings.livekit_internal_url,
            "segmentSeconds": int(self.settings.livekit_egress_segment_seconds),
        }
        if not self.settings.livekit_egress_enabled:
            result["ready"] = True
            return result
        try:
            import httpx

            livekit = httpx.get(self.settings.livekit_internal_url, timeout=2.0)
            egress = httpx.get(self.settings.livekit_egress_health_url, timeout=2.0)
            result["livekitReachable"] = livekit.status_code < 500
            result["egressReachable"] = egress.status_code < 500
            result["ready"] = bool(result["livekitReachable"] and result["egressReachable"])
        except Exception as exc:
            result.update({"ready": False, "error": str(exc)[:300]})
        return result
