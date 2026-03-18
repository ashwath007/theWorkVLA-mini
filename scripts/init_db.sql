-- India VLA Database Initialization
-- Enables pgvector extension for embedding similarity search

CREATE EXTENSION IF NOT EXISTS vector;

-- Session metadata table
CREATE TABLE IF NOT EXISTS sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  TEXT UNIQUE NOT NULL,
    data_dir    TEXT NOT NULL,
    start_time  DOUBLE PRECISION,
    end_time    DOUBLE PRECISION,
    video_frames INTEGER DEFAULT 0,
    duration_sec DOUBLE PRECISION DEFAULT 0,
    language    TEXT DEFAULT 'hi',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Episode metadata table
CREATE TABLE IF NOT EXISTS episodes (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_index        INTEGER NOT NULL,
    session_id           TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
    hdf5_path            TEXT NOT NULL,
    language_instruction TEXT,
    num_frames           INTEGER DEFAULT 0,
    duration_sec         DOUBLE PRECISION DEFAULT 0,
    motion_magnitude     FLOAT DEFAULT 0,
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast episode lookup by session
CREATE INDEX IF NOT EXISTS idx_episodes_session_id ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_index ON episodes(episode_index);

-- Training jobs table
CREATE TABLE IF NOT EXISTS training_jobs (
    job_id      TEXT PRIMARY KEY,
    status      TEXT DEFAULT 'queued',
    message     TEXT DEFAULT '',
    epoch       INTEGER DEFAULT 0,
    total_epochs INTEGER DEFAULT 0,
    train_loss  FLOAT,
    val_loss    FLOAT,
    config_json JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
