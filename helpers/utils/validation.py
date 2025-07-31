import uuid

import httpx


def validate_url(url_label: str, url: str | None) -> None:
    if not url:
        return

    if not url.startswith(("http://", "https://")):
        raise ValueError(f"{url_label} must start with http:// or https://")

    try:
        parsed = httpx.URL(url)
        if not parsed.host:
            raise ValueError("Invalid URL format")
    except Exception as e:
        raise ValueError(f"Invalid {url_label} format: {str(e)}")


def validate_uuid(uuid_type: str, uuid_str: str | None):
    if not uuid_str:
        return

    try:
        uuid.UUID(uuid_str)
    except ValueError:
        raise ValueError(f"{uuid_type} must be a valid UUID")
