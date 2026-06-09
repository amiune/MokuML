from __future__ import annotations

from urllib.parse import SplitResult, urlsplit, urlunsplit


def normalize_sqlalchemy_connection_string(connection_string: str) -> str:
    """Prefer installed SQLAlchemy drivers for supported URL aliases."""
    parsed = urlsplit(connection_string)
    if parsed.scheme in {"postgresql", "postgres"}:
        return urlunsplit(
            SplitResult(
                scheme="postgresql+psycopg",
                netloc=parsed.netloc,
                path=parsed.path,
                query=parsed.query,
                fragment=parsed.fragment,
            )
        )
    return connection_string

