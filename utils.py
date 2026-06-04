import base64
from typing import Optional, Tuple

from media_types import MediaType

PREFIX = "tmdb"


def _to_public_id(internal_id: str) -> str:
	encoded = base64.urlsafe_b64encode(internal_id.encode()).decode()
	return encoded.rstrip("=")

def _from_public_id(public_id: str) -> Optional[str]:
	try:
		padding = "=" * (-len(public_id) % 4)
		decoded = base64.urlsafe_b64decode(public_id + padding).decode()
		return decoded
	except Exception:
		return None


def encode_tmdb_id_into_my_id(
	tmdb_id: int,
	media_type: MediaType
) -> str:
	internal_id = f"{PREFIX}_{media_type}_{tmdb_id}"
	return _to_public_id(internal_id)

def decode_my_id_into_tmdb_id(
	my_id: str
) -> Optional[Tuple[int, MediaType]]:
	internal_id = _from_public_id(my_id)
	if not internal_id:
		return None
	parts = internal_id.split("_")
	if len(parts) != 3 or parts[0] != PREFIX:
		return None
	_, media_type, tmdb_id_raw = parts
	if media_type not in ("tv", "movie"):
		return None
	try:
		return int(tmdb_id_raw), media_type
	except ValueError:
		return None


def normalize_bot_item_id(raw: str) -> Optional[str]:
	"""Canonical plotwise public token (same as stored in user_titles.item_id)."""
	value = str(raw).strip()
	if not value:
		return None
	if decode_my_id_into_tmdb_id(value) is not None:
		return value
	if value.startswith(f"{PREFIX}_"):
		return _to_public_id(value)
	return None


def movie_id_to_public_map(forward: dict[str, int]) -> dict[str, str]:
	"""Invert token→MovieLens map to movieId (str)→public token for recommend output."""
	out: dict[str, str] = {}
	for token, movie_id in forward.items():
		public_id = normalize_bot_item_id(token)
		if public_id is None:
			continue
		out[str(int(movie_id))] = public_id
	return out


def item_id_to_public_id(
	raw: str,
	*,
	movie_id_to_public: dict[str, str] | None = None,
) -> Optional[str]:
	"""Catalog row id → plotwise public token (utils encoding), or None if unknown."""
	public_id = normalize_bot_item_id(raw)
	if public_id is not None:
		return public_id
	value = str(raw).strip()
	if value.isdigit() and movie_id_to_public is not None:
		return movie_id_to_public.get(value)
	return None
