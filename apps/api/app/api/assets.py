from fastapi import APIRouter, Depends, Header, HTTPException, Response

from app.core.security import require_api_auth
from app.services.media_store import MediaStore


router = APIRouter(prefix="/api/v1/assets", tags=["assets"])


@router.get("/{asset_id}/content", dependencies=[Depends(require_api_auth)])
def asset_content(asset_id: str, x_proctorcore_company: int | None = Header(None)) -> Response:
    try:
        store = MediaStore()
        asset = store._find_asset(asset_id)
        if asset is None or x_proctorcore_company is None or int(asset.get("companyId") or 0) != x_proctorcore_company:
            raise KeyError("asset_not_found")
        content, mime_type = store.get_asset_content(asset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="asset_not_found") from exc
    return Response(
        content=content,
        media_type=mime_type,
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.delete("/{asset_id}", dependencies=[Depends(require_api_auth)])
def delete_asset(asset_id: str, x_proctorcore_company: int | None = Header(None)) -> dict:
    store = MediaStore()
    asset = store._find_asset(asset_id)
    if asset is None or x_proctorcore_company is None or int(asset.get("companyId") or 0) != x_proctorcore_company:
        raise HTTPException(status_code=404, detail="asset_not_found")
    return {"ok": True, "deleted": store.delete_asset(asset_id), "deletedAt": int(__import__("time").time())}
