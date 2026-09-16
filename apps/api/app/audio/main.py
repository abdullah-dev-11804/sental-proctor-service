from __future__ import annotations

import asyncio
import json
import signal
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings, get_settings
from app.services.audio_analysis import (
    AudioEventEngine,
    AudioPolicy,
    SileroVad,
    SpeechBrainSpeakerEncoder,
    audio_model_status,
)
from app.services.media_store import MediaStore
from app.services.state_store import StateStore


class AudioSupervisor:
    """Discovers active sessions and subscribes to their candidate audio tracks."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.state = StateStore(self.settings)
        self.tasks: dict[str, asyncio.Task] = {}
        self.stopping = asyncio.Event()
        self.vad: SileroVad | None = None
        self.speaker: SpeechBrainSpeakerEncoder | None = None
        self.model_state: dict[str, Any] = {}

    async def run(self) -> None:
        self._load_models()
        while not self.stopping.is_set():
            await self._reconcile_sessions()
            self._write_health()
            try:
                await asyncio.wait_for(
                    self.stopping.wait(),
                    timeout=max(0.5, float(self.settings.audio_supervisor_poll_seconds)),
                )
            except asyncio.TimeoutError:
                pass
        await self._stop_tasks()

    def stop(self) -> None:
        self.stopping.set()

    def _load_models(self) -> None:
        self.model_state = audio_model_status(self.settings)
        if not self.settings.audio_analysis_enabled:
            return
        try:
            self.vad = SileroVad(self.settings)
            self.speaker = SpeechBrainSpeakerEncoder(self.settings)
            self.model_state["available"] = True
            self.model_state["modelsLoaded"] = True
        except Exception as exc:
            self.vad = None
            self.speaker = None
            self.model_state["available"] = False
            self.model_state["modelsLoaded"] = False
            self.model_state["error"] = str(exc)[:500]

    async def _reconcile_sessions(self) -> None:
        finished = [session_id for session_id, task in self.tasks.items() if task.done()]
        for session_id in finished:
            task = self.tasks.pop(session_id)
            with suppress(asyncio.CancelledError):
                exception = task.exception()
                if exception:
                    self._mark_session_state(session_id, "failed", str(exception))

        for session_id in self.state.list_session_ids():
            if session_id in self.tasks:
                continue
            try:
                session = self.state.get_session(session_id)
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
            policy = AudioPolicy.from_session(session, self.settings)
            recording = session.get("recording") if isinstance(session.get("recording"), dict) else {}
            requested = bool((session.get("audioAnalysis") or {}).get("enabled"))
            if requested and not self.settings.audio_analysis_enabled:
                self._mark_session_state(session_id, "degraded", "audio_analysis_disabled_on_server")
                continue
            if requested and not self.model_state.get("available"):
                self._mark_session_state(
                    session_id,
                    "degraded",
                    str(self.model_state.get("error") or "audio_models_unavailable"),
                )
                continue
            if policy.enabled and recording.get("state") == "active" and session.get("status") == "active":
                self.tasks[session_id] = asyncio.create_task(
                    self._monitor_session(session_id),
                    name=f"audio-{session_id}",
                )

    async def _monitor_session(self, session_id: str) -> None:
        from livekit import api, rtc

        session = self.state.get_session(session_id)
        policy = AudioPolicy.from_session(session, self.settings)
        if not policy.enabled or self.vad is None:
            return
        # Silero carries recurrent state, so each microphone stream requires an
        # independent model wrapper even though the tiny ONNX weights are shared
        # by the operating-system page cache.
        engine = AudioEventEngine(policy, SileroVad(self.settings), self.speaker)
        room_name = str(session.get("roomId") or "")
        candidate_identity = str(session.get("participantIdentity") or f"user-{session.get('userId') or 'unknown'}")
        token = (
            api.AccessToken(self.settings.livekit_api_key, self.settings.livekit_api_secret)
            .with_identity(f"audio-analyzer-{session_id[:48]}")
            .with_name("ProctorCore audio analyzer")
            .with_grants(api.VideoGrants(room_join=True, room=room_name, can_publish=False, can_subscribe=True))
            .to_jwt()
        )
        room = rtc.Room()
        candidate_track: asyncio.Future = asyncio.get_running_loop().create_future()

        @room.on("track_subscribed")
        def on_track_subscribed(track, _publication, participant) -> None:
            if participant.identity == candidate_identity \
                    and track.kind == rtc.TrackKind.KIND_AUDIO \
                    and not candidate_track.done():
                candidate_track.set_result(track)

        self._mark_session_state(session_id, "connecting")
        try:
            await room.connect(self.settings.livekit_internal_url, token, options=rtc.RoomOptions(auto_subscribe=True))
            track = await asyncio.wait_for(candidate_track, timeout=30)
            self._mark_session_state(session_id, "active")
            stream = rtc.AudioStream(
                track,
                capacity=100,
                sample_rate=16000,
                num_channels=1,
                frame_size_ms=32,
            )
            async for event in stream:
                current = self.state.get_session(session_id)
                recording = current.get("recording") if isinstance(current.get("recording"), dict) else {}
                if current.get("status") != "active" or recording.get("state") != "active":
                    break
                frame = event.frame
                samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
                ended_at = time.time()
                started_at = ended_at - (samples.size / 16000.0)
                for violation in engine.process(samples, started_at):
                    self._record_event(current, violation)
            current = self.state.get_session(session_id)
            for violation in engine.flush(time.time()):
                self._record_event(current, violation)
            self._mark_session_state(session_id, "stopped")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._mark_session_state(session_id, "failed", str(exc))
            raise
        finally:
            await room.disconnect()

    def _record_event(self, session: dict[str, Any], event: dict[str, Any]) -> None:
        session_id = str(session["id"])
        recording = session.get("recording") if isinstance(session.get("recording"), dict) else {}
        event_type = str(event["type"])
        descriptions = {
            "background_noise": "Background noise detected.",
            "speech_detected": "Human speech was detected.",
            "second_voice_detected": "Additional speaker detected.",
            "possible_prompting": "Possible prompting / sustained speech activity detected. Review required.",
        }
        severities = {
            "background_noise": "notice",
            "speech_detected": "warning",
            "second_voice_detected": "high",
            "possible_prompting": "high",
        }
        started_at = int(event["startedAt"])
        metadata = dict(event.get("metadata") or {})
        metadata.update({
            "eventStartTimestamp": started_at,
            "eventEndTimestamp": int(event["endedAt"]),
            "durationMs": int(event["durationMs"]),
            "recordingReference": {
                "provider": str(recording.get("provider") or "livekit"),
                "roomId": session.get("roomId"),
                "recordingSegment": int(recording.get("currentSegment") or 1),
                "startTimestamp": started_at,
                "endTimestamp": int(event["endedAt"]),
            },
            "containsTranscript": False,
        })
        MediaStore().record_violation({
            "sessionId": session_id,
            "companyId": int(session.get("companyId") or 0),
            "moodleSessionId": session.get("moodleSessionId"),
            "attemptId": session.get("attemptId"),
            "userId": session.get("userId"),
            "violationId": f"audio-{session_id[:48]}-{event_type}-{started_at}-{int(event['endedAt'])}",
            "violationType": event_type,
            "severity": severities[event_type],
            "confidence": event.get("confidence"),
            "occurredAt": started_at,
            "endedAt": int(event["endedAt"]),
            "durationMs": int(event["durationMs"]),
            "description": descriptions[event_type],
            "metadata": metadata,
            "source": "audio_analysis",
        })

    def _mark_session_state(self, session_id: str, status: str, error: str = "") -> None:
        try:
            with self.state.session_lock(session_id):
                session = self.state.get_session(session_id)
                previous = session.get("audioAnalysisState") \
                    if isinstance(session.get("audioAnalysisState"), dict) else {}
                if previous.get("status") == status \
                        and str(previous.get("error") or "") == (error[:500] or "") \
                        and time.time() - int(previous.get("updatedAt") or 0) < 15:
                    return
                session["audioAnalysisState"] = {
                    "status": status,
                    "updatedAt": int(time.time()),
                    "sampleRate": 16000,
                    "error": error[:500] or None,
                }
                self.state.save_session(session)
        except Exception:
            pass

    def _write_health(self) -> None:
        payload = {
            "updatedAt": int(time.time()),
            "ready": bool(
                not self.settings.audio_analysis_enabled
                or self.model_state.get("available")
            ),
            "enabled": bool(self.settings.audio_analysis_enabled),
            "activeSessions": len(self.tasks),
            "sampleRate": 16000,
            "models": self.model_state,
        }
        encoded = json.dumps(payload, sort_keys=True)
        try:
            self.state.redis.set("proctorcore:audio:health", encoded, ex=60)
        except Exception:
            health_path = self.settings.local_storage_root / "audio-analysis-health.json"
            health_path.write_text(encoded, encoding="utf-8")

    async def _stop_tasks(self) -> None:
        for task in self.tasks.values():
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        self.tasks.clear()


def main() -> None:
    supervisor = AudioSupervisor()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for signame in (signal.SIGTERM, signal.SIGINT):
        with suppress(NotImplementedError):
            loop.add_signal_handler(signame, supervisor.stop)
    try:
        loop.run_until_complete(supervisor.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
