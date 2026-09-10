import hmac

from fastapi import Header, HTTPException, status

from app.core.config import get_settings


def require_api_auth(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    settings = get_settings()
    expected = settings.api_shared_secret

    token = ""
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer":
            token = value.strip()
    if not token and x_api_key:
        token = x_api_key.strip()

    if not expected or not hmac.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "unauthorized", "message": "Missing or invalid API token."},
        )


def require_company_scope(company_id: int, header_company_id: int | None) -> None:
    """Fails closed when a trusted caller omits or forges its tenant header."""
    if header_company_id is None or int(header_company_id) != int(company_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="resource_not_found")
