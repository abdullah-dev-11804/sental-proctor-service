from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.core.security import require_api_auth
from app.services.media_store import MediaStore


router = APIRouter(prefix="/api/v1/assets", tags=["assets"])


@router.get("/{asset_id}/content", dependencies=[Depends(require_api_auth)])
def asset_content(asset_id: str) -> FileResponse:
    try:
        path, mime_type = MediaStore().get_asset_content(asset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="asset_not_found") from exc
    return FileResponse(path, media_type=mime_type, filename=path.name)


@router.delete("/{asset_id}", dependencies=[Depends(require_api_auth)])
def delete_asset(asset_id: str) -> dict:
    return {"ok": True, "deleted": MediaStore().delete_asset(asset_id)}
