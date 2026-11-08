"""Move only known legacy uploads into ACE's accepted private temp pool."""
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

_NAME = re.compile(r"^[a-f0-9]{32}\.(mp3|wav|flac|ogg|m4a|opus)$")


def reference_pool():
    pool = Path(tempfile.gettempdir()) / f"oneflow-acestep-refs-{os.getuid()}"
    pool.mkdir(mode=0o700, exist_ok=True)
    if pool.is_symlink() or pool.stat().st_uid != os.getuid():
        raise ValueError("unsafe voice reference upload directory")
    pool.chmod(0o700)
    return pool


def migrate_uploaded_reference(value, legacy, pool):
    if not isinstance(value, str) or not value.startswith(str(legacy) + "/"):
        return value  # Unrelated paths still go through ACE's original validator.
    source = Path(value)
    if not _NAME.fullmatch(source.name) or source.parent != legacy or source.is_symlink():
        raise ValueError("invalid legacy voice reference")
    if not source.is_file() or source.stat().st_uid != os.getuid() or not 1024 <= source.stat().st_size <= 30 * 1024 * 1024:
        raise ValueError("legacy voice reference missing or invalid")
    target = pool / source.name
    if target.is_symlink():
        raise ValueError("invalid migrated voice reference")
    if target.exists():
        if not target.is_file() or target.stat().st_uid != os.getuid() or target.stat().st_size != source.stat().st_size:
            raise ValueError("migrated voice reference conflict")
        return str(target)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=pool, prefix="migrate_", delete=False) as output:
            temporary = Path(output.name)
            with source.open("rb") as original:
                shutil.copyfileobj(original, output)
        os.replace(temporary, target)
        return str(target)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


class LegacyVoiceReferenceMiddleware:
    def __init__(self, app, legacy=None, pool=None):
        self.app = app
        self.legacy = Path(legacy) if legacy else Path.home() / ".cache/acestep/tmp/refs"
        self.pool = Path(pool) if pool else reference_pool()

    async def __call__(self, scope, receive, send):
        headers = dict(scope.get("headers", []))
        if scope["type"] != "http" or scope.get("path") != "/release_task" or scope.get("method") != "POST" or not headers.get(b"content-type", b"").startswith(b"application/json"):
            return await self.app(scope, receive, send)
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > 2 * 1024 * 1024:
                return await self.error(send, "release payload too large", 413)
            chunks.append(chunk)
            if not message.get("more_body"):
                break
        body = b"".join(chunks)
        try:
            payload = json.loads(body)
            if isinstance(payload, dict) and payload.get("reference_audio_path"):
                original = payload["reference_audio_path"]
                mapped = migrate_uploaded_reference(original, self.legacy, self.pool)
                if mapped != original:
                    payload["reference_audio_path"] = mapped
                    body = json.dumps(payload).encode()
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass  # Let ACE report its normal invalid-JSON error.
        except ValueError as error:
            return await self.error(send, str(error), 400)
        delivered = False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()
        updated = dict(scope)
        updated["headers"] = [(k,v) for k,v in scope.get("headers",[]) if k != b"content-length"] + [(b"content-length",str(len(body)).encode())]
        await self.app(updated, replay, send)

    @staticmethod
    async def error(send, message, status):
        body = json.dumps({"detail": message}).encode()
        await send({"type":"http.response.start","status":status,"headers":[(b"content-type",b"application/json")]})
        await send({"type":"http.response.body","body":body})
