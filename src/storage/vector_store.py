"""
VectorStore: pgvector + PostGIS semantic and geospatial episode search.

Provides:
- Episode insertion / upsert
- Cosine-similarity semantic search (pgvector)
- Geospatial proximity search (PostGIS ST_DWithin)
- Full-text ILIKE search on language instructions
- Per-session episode listing and aggregate statistics

Falls back transparently to an in-memory MockVectorStore when asyncpg /
psycopg2 are not installed, so the codebase runs without a database for
local development and testing.

Also includes SimpleEmbedder for prototype-quality text and image embeddings
that require no model downloads.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional database backend detection ───────────────────────────────────────

_HAS_SQLALCHEMY = False
_HAS_ASYNCPG    = False

try:
    import sqlalchemy  # type: ignore
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncConnection  # type: ignore
    from sqlalchemy import text  # type: ignore
    _HAS_SQLALCHEMY = True
except ImportError:
    logger.warning("SQLAlchemy not installed; VectorStore will use MockVectorStore.")

try:
    import asyncpg  # type: ignore  # noqa: F401
    _HAS_ASYNCPG = True
except ImportError:
    pass

_USE_DB = _HAS_SQLALCHEMY and _HAS_ASYNCPG

# ── Schema DDL ────────────────────────────────────────────────────────────────

_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS episodes (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id               TEXT NOT NULL,
    episode_index            INTEGER NOT NULL,
    hdf5_path                TEXT NOT NULL,
    language_instruction     TEXT,
    language_instruction_hi  TEXT,
    scenario_type            TEXT,
    frame_count              INTEGER,
    duration_sec             FLOAT,
    created_at               TIMESTAMP DEFAULT NOW(),
    start_location           GEOGRAPHY(POINT, 4326),
    end_location             GEOGRAPHY(POINT, 4326),
    route_line               GEOGRAPHY(LINESTRING, 4326),
    instruction_embedding    VECTOR(512),
    visual_embedding         VECTOR(512)
);

CREATE INDEX IF NOT EXISTS idx_episodes_session
    ON episodes(session_id);

CREATE INDEX IF NOT EXISTS idx_episodes_scenario
    ON episodes(scenario_type);

CREATE INDEX IF NOT EXISTS idx_instruction_embedding
    ON episodes USING ivfflat (instruction_embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_start_location
    ON episodes USING GIST(start_location);
"""


# ── VectorStore ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    Async PostgreSQL-backed episode store with pgvector and PostGIS.

    Parameters
    ----------
    db_url : str
        SQLAlchemy-compatible async database URL, e.g.
        ``postgresql+asyncpg://user:pass@host/dbname``.
    """

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self._engine = None

        if not _USE_DB:
            logger.warning(
                "asyncpg/SQLAlchemy unavailable — VectorStore running in mock mode."
            )
        else:
            try:
                self._engine = create_async_engine(db_url, echo=False, pool_pre_ping=True)
            except Exception as exc:
                logger.error("Failed to create database engine: %s", exc)
                self._engine = None

    # ── Connection context manager ────────────────────────────────────────────

    @asynccontextmanager
    async def connect(self) -> AsyncGenerator[AsyncConnection, None]:
        """Async context manager yielding a live database connection.

        Raises RuntimeError if the database engine is not available.
        """
        if self._engine is None:
            raise RuntimeError(
                "Database engine not initialised. Check db_url and asyncpg installation."
            )
        async with self._engine.connect() as conn:
            yield conn

    # ── Schema init ───────────────────────────────────────────────────────────

    async def init_schema(self) -> None:
        """Create extensions, tables, and indexes if they do not exist."""
        if self._engine is None:
            logger.warning("init_schema: no engine available, skipping.")
            return

        async with self.connect() as conn:
            # Run each DDL statement separately (CREATE EXTENSION must be isolated)
            for statement in _DDL.split(";"):
                stmt = statement.strip()
                if stmt:
                    try:
                        await conn.execute(text(stmt))
                    except Exception as exc:
                        logger.warning("DDL statement warning: %s", exc)
            await conn.commit()

        logger.info("VectorStore schema initialised.")

    # ── Insert / upsert ───────────────────────────────────────────────────────

    async def insert_episode(self, episode_meta: Dict[str, Any]) -> str:
        """
        Insert or upsert an episode record.

        Parameters
        ----------
        episode_meta : dict
            Recognised keys (all optional except session_id, hdf5_path):
                session_id, episode_index, hdf5_path,
                language_instruction, language_instruction_hi,
                scenario_type, frame_count, duration_sec,
                start_lat, start_lon, end_lat, end_lon,
                route_coords (list of [lon, lat] pairs for LINESTRING),
                instruction_embedding (np.ndarray or list, 512-d),
                visual_embedding      (np.ndarray or list, 512-d)

        Returns
        -------
        str  — the UUID of the inserted / updated episode row.
        """
        if self._engine is None:
            raise RuntimeError("Database engine not available.")

        ep_id = str(uuid.uuid4())
        now   = datetime.utcnow()

        # Geography helpers
        start_loc  = _point_wkt(episode_meta.get("start_lat"), episode_meta.get("start_lon"))
        end_loc    = _point_wkt(episode_meta.get("end_lat"),   episode_meta.get("end_lon"))
        route_line = _linestring_wkt(episode_meta.get("route_coords"))

        instr_emb = _embed_to_pgvector(episode_meta.get("instruction_embedding"))
        vis_emb   = _embed_to_pgvector(episode_meta.get("visual_embedding"))

        sql = text("""
            INSERT INTO episodes (
                id, session_id, episode_index, hdf5_path,
                language_instruction, language_instruction_hi,
                scenario_type, frame_count, duration_sec, created_at,
                start_location, end_location, route_line,
                instruction_embedding, visual_embedding
            ) VALUES (
                :id, :session_id, :episode_index, :hdf5_path,
                :language_instruction, :language_instruction_hi,
                :scenario_type, :frame_count, :duration_sec, :created_at,
                ST_GeogFromText(:start_location),
                ST_GeogFromText(:end_location),
                ST_GeogFromText(:route_line),
                :instruction_embedding::vector,
                :visual_embedding::vector
            )
            ON CONFLICT (id) DO UPDATE SET
                language_instruction    = EXCLUDED.language_instruction,
                language_instruction_hi = EXCLUDED.language_instruction_hi,
                instruction_embedding   = EXCLUDED.instruction_embedding,
                visual_embedding        = EXCLUDED.visual_embedding
        """)

        async with self.connect() as conn:
            await conn.execute(sql, {
                "id":                      ep_id,
                "session_id":              episode_meta.get("session_id", ""),
                "episode_index":           int(episode_meta.get("episode_index", 0)),
                "hdf5_path":               episode_meta.get("hdf5_path", ""),
                "language_instruction":    episode_meta.get("language_instruction"),
                "language_instruction_hi": episode_meta.get("language_instruction_hi"),
                "scenario_type":           episode_meta.get("scenario_type"),
                "frame_count":             episode_meta.get("frame_count"),
                "duration_sec":            episode_meta.get("duration_sec"),
                "created_at":              now,
                "start_location":          start_loc,
                "end_location":            end_loc,
                "route_line":              route_line,
                "instruction_embedding":   instr_emb,
                "visual_embedding":        vis_emb,
            })
            await conn.commit()

        logger.debug("Episode inserted: %s", ep_id)
        return ep_id

    # ── Semantic search ───────────────────────────────────────────────────────

    async def search_semantic(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        scenario_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Cosine-similarity search via pgvector.

        Parameters
        ----------
        query_embedding : np.ndarray  shape (512,)
        top_k : int
        scenario_type : str, optional — filter by scenario

        Returns
        -------
        list of dicts ordered by similarity (highest first).
        """
        if self._engine is None:
            raise RuntimeError("Database engine not available.")

        vec_str = _embed_to_pgvector(query_embedding)
        where   = "WHERE 1=1"
        params: Dict[str, Any] = {
            "vec":   vec_str,
            "top_k": top_k,
        }

        if scenario_type:
            where += " AND scenario_type = :scenario_type"
            params["scenario_type"] = scenario_type

        sql = text(f"""
            SELECT id, session_id, episode_index, hdf5_path,
                   language_instruction, scenario_type, frame_count, duration_sec,
                   1 - (instruction_embedding <=> :vec::vector) AS similarity
            FROM episodes
            {where}
            ORDER BY instruction_embedding <=> :vec::vector
            LIMIT :top_k
        """)

        async with self.connect() as conn:
            result = await conn.execute(sql, params)
            rows = result.mappings().all()

        return [dict(r) for r in rows]

    # ── Geo search ────────────────────────────────────────────────────────────

    async def search_geo(
        self,
        lat: float,
        lon: float,
        radius_km: float = 1.0,
        scenario_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        PostGIS ST_DWithin proximity search.

        Parameters
        ----------
        lat, lon : float  — query centre point
        radius_km : float  — search radius in kilometres
        scenario_type : str, optional

        Returns
        -------
        list of episode dicts ordered by distance.
        """
        if self._engine is None:
            raise RuntimeError("Database engine not available.")

        radius_m = radius_km * 1000.0
        where    = "WHERE ST_DWithin(start_location, ST_GeogFromText(:point), :radius)"
        params: Dict[str, Any] = {
            "point":  f"POINT({lon} {lat})",
            "radius": radius_m,
        }

        if scenario_type:
            where += " AND scenario_type = :scenario_type"
            params["scenario_type"] = scenario_type

        sql = text(f"""
            SELECT id, session_id, episode_index, hdf5_path,
                   language_instruction, scenario_type, frame_count, duration_sec,
                   ST_Distance(start_location, ST_GeogFromText(:point)) AS distance_m
            FROM episodes
            {where}
            ORDER BY distance_m ASC
        """)

        async with self.connect() as conn:
            result = await conn.execute(sql, params)
            rows = result.mappings().all()

        return [dict(r) for r in rows]

    # ── Text search ───────────────────────────────────────────────────────────

    async def search_text(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Case-insensitive ILIKE search on language_instruction.

        Parameters
        ----------
        query : str  — search term (may include SQL wildcards)
        top_k : int

        Returns
        -------
        list of episode dicts.
        """
        if self._engine is None:
            raise RuntimeError("Database engine not available.")

        # Escape existing wildcard chars and wrap in %…%
        safe_query = query.replace("%", r"\%").replace("_", r"\_")
        pattern = f"%{safe_query}%"

        sql = text("""
            SELECT id, session_id, episode_index, hdf5_path,
                   language_instruction, scenario_type, frame_count, duration_sec
            FROM episodes
            WHERE language_instruction ILIKE :pattern
               OR language_instruction_hi ILIKE :pattern
            LIMIT :top_k
        """)

        async with self.connect() as conn:
            result = await conn.execute(sql, {"pattern": pattern, "top_k": top_k})
            rows = result.mappings().all()

        return [dict(r) for r in rows]

    # ── Session episodes ──────────────────────────────────────────────────────

    async def get_session_episodes(self, session_id: str) -> List[Dict[str, Any]]:
        """Return all episodes for a session, ordered by episode_index."""
        if self._engine is None:
            raise RuntimeError("Database engine not available.")

        sql = text("""
            SELECT id, session_id, episode_index, hdf5_path,
                   language_instruction, scenario_type, frame_count, duration_sec
            FROM episodes
            WHERE session_id = :session_id
            ORDER BY episode_index ASC
        """)

        async with self.connect() as conn:
            result = await conn.execute(sql, {"session_id": session_id})
            rows = result.mappings().all()

        return [dict(r) for r in rows]

    # ── Stats ─────────────────────────────────────────────────────────────────

    async def get_stats(self) -> Dict[str, Any]:
        """
        Return aggregate dataset statistics.

        Returns
        -------
        dict with keys:
            total_episodes, total_sessions, by_scenario (dict), total_duration_hours
        """
        if self._engine is None:
            raise RuntimeError("Database engine not available.")

        sql = text("""
            SELECT
                COUNT(*)                       AS total_episodes,
                COUNT(DISTINCT session_id)     AS total_sessions,
                COALESCE(SUM(duration_sec), 0) AS total_duration_sec,
                scenario_type,
                COUNT(*) FILTER (WHERE scenario_type IS NOT NULL) AS scenario_count
            FROM episodes
            GROUP BY ROLLUP(scenario_type)
        """)

        async with self.connect() as conn:
            result = await conn.execute(sql)
            rows = result.mappings().all()

        total_episodes  = 0
        total_sessions  = 0
        total_duration  = 0.0
        by_scenario: Dict[str, int] = {}

        for row in rows:
            row = dict(row)
            if row.get("scenario_type") is None:
                # ROLLUP grand total row
                total_episodes = int(row.get("total_episodes", 0))
                total_sessions = int(row.get("total_sessions", 0))
                total_duration = float(row.get("total_duration_sec", 0))
            else:
                by_scenario[row["scenario_type"]] = int(row.get("total_episodes", 0))

        return {
            "total_episodes":       total_episodes,
            "total_sessions":       total_sessions,
            "by_scenario":          by_scenario,
            "total_duration_hours": round(total_duration / 3600, 2),
        }

    # ── Delete ────────────────────────────────────────────────────────────────

    async def delete_episode(self, episode_id: str) -> bool:
        """Delete a single episode by UUID. Returns True if a row was deleted."""
        if self._engine is None:
            raise RuntimeError("Database engine not available.")

        sql = text("DELETE FROM episodes WHERE id = :id")
        async with self.connect() as conn:
            result = await conn.execute(sql, {"id": episode_id})
            await conn.commit()
        return result.rowcount > 0

    async def delete_session(self, session_id: str) -> int:
        """Delete all episodes for a session. Returns number of rows deleted."""
        if self._engine is None:
            raise RuntimeError("Database engine not available.")

        sql = text("DELETE FROM episodes WHERE session_id = :session_id")
        async with self.connect() as conn:
            result = await conn.execute(sql, {"session_id": session_id})
            await conn.commit()
        return result.rowcount


# ── MockVectorStore ───────────────────────────────────────────────────────────

class MockVectorStore:
    """
    In-memory stand-in for VectorStore used when no database is available.

    All data is stored in a list of dicts; similarity is computed with numpy.
    Suitable for unit tests and local development only.
    """

    def __init__(self) -> None:
        self._episodes: List[Dict[str, Any]] = []

    async def init_schema(self) -> None:
        logger.info("MockVectorStore: no-op init_schema")

    @asynccontextmanager
    async def connect(self) -> AsyncGenerator[None, None]:
        yield  # no-op — data lives in memory

    async def insert_episode(self, episode_meta: Dict[str, Any]) -> str:
        ep_id = str(uuid.uuid4())
        record = dict(episode_meta)
        record["id"]         = ep_id
        record["created_at"] = datetime.utcnow().isoformat()
        self._episodes.append(record)
        logger.debug("MockVectorStore: inserted episode %s", ep_id)
        return ep_id

    async def search_semantic(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        scenario_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        q = query_embedding.astype(np.float32).flatten()
        for ep in self._episodes:
            if scenario_type and ep.get("scenario_type") != scenario_type:
                continue
            emb = ep.get("instruction_embedding")
            if emb is None:
                continue
            v = np.asarray(emb, dtype=np.float32).flatten()
            # Cosine similarity
            denom = (np.linalg.norm(q) * np.linalg.norm(v))
            sim = float(np.dot(q, v) / denom) if denom > 0 else 0.0
            results.append({**ep, "similarity": sim})

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    async def search_geo(
        self,
        lat: float,
        lon: float,
        radius_km: float = 1.0,
        scenario_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for ep in self._episodes:
            if scenario_type and ep.get("scenario_type") != scenario_type:
                continue
            slat = ep.get("start_lat")
            slon = ep.get("start_lon")
            if slat is None or slon is None:
                continue
            dist_m = _haversine_mock(lat, lon, float(slat), float(slon))
            if dist_m <= radius_km * 1000:
                results.append({**ep, "distance_m": dist_m})
        results.sort(key=lambda x: x["distance_m"])
        return results

    async def search_text(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        q = query.lower()
        results = [
            ep for ep in self._episodes
            if q in (ep.get("language_instruction") or "").lower()
            or q in (ep.get("language_instruction_hi") or "").lower()
        ]
        return results[:top_k]

    async def get_session_episodes(self, session_id: str) -> List[Dict[str, Any]]:
        return sorted(
            [ep for ep in self._episodes if ep.get("session_id") == session_id],
            key=lambda x: x.get("episode_index", 0),
        )

    async def get_stats(self) -> Dict[str, Any]:
        total    = len(self._episodes)
        sessions = len({ep.get("session_id") for ep in self._episodes})
        duration = sum(float(ep.get("duration_sec") or 0) for ep in self._episodes)
        scenario_counts: Counter = Counter(
            ep.get("scenario_type") for ep in self._episodes if ep.get("scenario_type")
        )
        return {
            "total_episodes":       total,
            "total_sessions":       sessions,
            "by_scenario":          dict(scenario_counts),
            "total_duration_hours": round(duration / 3600, 2),
        }

    async def delete_episode(self, episode_id: str) -> bool:
        before = len(self._episodes)
        self._episodes = [ep for ep in self._episodes if ep.get("id") != episode_id]
        return len(self._episodes) < before

    async def delete_session(self, session_id: str) -> int:
        before = len(self._episodes)
        self._episodes = [ep for ep in self._episodes if ep.get("session_id") != session_id]
        return before - len(self._episodes)


# ── SimpleEmbedder ────────────────────────────────────────────────────────────

# Vocabulary of common robotics / delivery / domestic task words used for TF-IDF
# style bag-of-words text embedding.
_VOCAB_WORDS = [
    # Actions
    "pick", "place", "carry", "deliver", "move", "push", "pull", "open", "close",
    "turn", "fold", "stack", "sort", "pour", "lift", "put", "take", "bring",
    "drop", "grab", "hold", "press", "switch", "scan", "pack", "unpack",
    # Objects
    "box", "bag", "bottle", "cup", "plate", "table", "chair", "door", "shelf",
    "drawer", "container", "item", "product", "package", "parcel", "food",
    "object", "tool", "button", "handle", "key", "phone", "book",
    # Locations
    "kitchen", "warehouse", "delivery", "office", "room", "counter", "floor",
    "wall", "aisle", "corridor", "entrance", "exit", "left", "right", "forward",
    "back", "up", "down", "inside", "outside", "near", "far",
    # Context
    "robot", "human", "task", "action", "navigation", "manipulation", "grasp",
    "assembly", "inspection", "cleaning", "driving", "walking", "running",
    "sitting", "standing", "interact", "avoid", "detect", "search",
    # Hindi transliterations (common task words)
    "utha", "rakh", "le", "jao", "lao", "karo", "band", "kholo", "daal",
    "nikal", "pakar", "chalo", "ruko", "dhundho", "dena", "lena",
]

# Build vocab lookup: word → index
_VOCAB: Dict[str, int] = {w: i for i, w in enumerate(_VOCAB_WORDS)}
_VOCAB_SIZE = len(_VOCAB_WORDS)


class SimpleEmbedder:
    """
    Prototype-quality text and image embedder that requires no model downloads.

    Text embeddings are 512-dimensional bag-of-words TF-IDF style vectors
    built from a fixed robotics / delivery vocabulary.  The first *VOCAB_SIZE*
    dimensions correspond to vocabulary term frequencies; the remaining
    dimensions are populated with character-level hash features to fill the
    512-d space.

    Image embeddings are mean-pooled pixel feature vectors (for prototyping
    only — replace with CLIP for production).
    """

    _DIM = 512

    # ── Text embedding ────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> np.ndarray:
        """
        Return a 512-dimensional float32 embedding for a text string.

        The embedding combines:
        - Vocabulary term-frequency features (first _VOCAB_SIZE dims)
        - Character n-gram hash features (remaining dims) for out-of-vocab coverage

        Parameters
        ----------
        text : str

        Returns
        -------
        np.ndarray  shape (512,), dtype float32, L2-normalised.
        """
        vec = np.zeros(self._DIM, dtype=np.float32)

        if not text:
            return vec

        # Tokenise (lowercase, split on non-alphanumeric)
        tokens = re.findall(r"[a-zA-Z\u0900-\u097F]+", text.lower())

        # ── Vocabulary term frequency ─────────────────────────────────────────
        if tokens:
            for tok in tokens:
                idx = _VOCAB.get(tok)
                if idx is not None and idx < self._DIM:
                    vec[idx] += 1.0

        # ── Character bigram hash features ───────────────────────────────────
        # Fill remaining dimensions with hash-based features
        offset = _VOCAB_SIZE
        capacity = self._DIM - offset
        if capacity > 0 and tokens:
            for tok in tokens:
                for j in range(len(tok) - 1):
                    bigram = tok[j : j + 2]
                    h = int(hashlib.md5(bigram.encode()).hexdigest(), 16)
                    slot = offset + (h % capacity)
                    vec[slot] += 1.0

        # L2 normalise
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec

    # ── Image (placeholder) embedding ────────────────────────────────────────

    def embed_image_placeholder(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute a 512-dimensional prototype embedding from a video frame.

        Uses mean-pooled pixel values across colour channels and spatial
        blocks.  This is a placeholder — for production use CLIP.

        Parameters
        ----------
        frame : np.ndarray  shape (H, W, C) or (H, W), dtype uint8 or float32

        Returns
        -------
        np.ndarray  shape (512,), dtype float32, L2-normalised.
        """
        if frame.ndim == 2:
            frame = frame[:, :, np.newaxis]

        frame_f = frame.astype(np.float32) / 255.0
        H, W, C = frame_f.shape

        # Divide image into a grid of blocks and compute per-block channel means
        n_blocks_h = 8
        n_blocks_w = 8
        features: List[float] = []

        bh = max(1, H // n_blocks_h)
        bw = max(1, W // n_blocks_w)

        for brow in range(n_blocks_h):
            for bcol in range(n_blocks_w):
                r0, r1 = brow * bh, min((brow + 1) * bh, H)
                c0, c1 = bcol * bw, min((bcol + 1) * bw, W)
                block  = frame_f[r0:r1, c0:c1, :]
                # Mean per channel
                for ch in range(min(C, 3)):
                    features.append(float(block[:, :, ch].mean()))
                # Std per channel
                for ch in range(min(C, 3)):
                    features.append(float(block[:, :, ch].std()))

        vec = np.array(features, dtype=np.float32)

        # Pad or truncate to _DIM
        if len(vec) < self._DIM:
            vec = np.pad(vec, (0, self._DIM - len(vec)))
        else:
            vec = vec[: self._DIM]

        # L2 normalise
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec


# ── Private helpers ───────────────────────────────────────────────────────────

def _point_wkt(lat: Optional[float], lon: Optional[float]) -> Optional[str]:
    """Return a WKT POINT string or None if coordinates are missing."""
    if lat is None or lon is None:
        return None
    return f"POINT({lon} {lat})"


def _linestring_wkt(coords: Optional[List]) -> Optional[str]:
    """
    Return a WKT LINESTRING string from a list of [lon, lat] pairs, or None.
    """
    if not coords or len(coords) < 2:
        return None
    pairs = " ".join(f"{c[0]} {c[1]}" for c in coords if len(c) >= 2)
    return f"LINESTRING({pairs})"


def _embed_to_pgvector(emb: Optional[Any]) -> Optional[str]:
    """
    Convert a numpy array or list to a pgvector-compatible string '[x,y,z,…]'.
    Returns None if emb is None.
    """
    if emb is None:
        return None
    arr = np.asarray(emb, dtype=np.float32).flatten()
    return "[" + ",".join(f"{v:.8f}" for v in arr) + "]"


def _haversine_mock(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in metres (used by MockVectorStore)."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))
