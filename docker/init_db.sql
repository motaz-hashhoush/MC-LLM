-- ============================================================================
-- MC-LLM  —  Database Initialisation Script
-- ============================================================================
-- Run this against your local PostgreSQL instance ONCE before starting the app.
--
-- Usage (psql):
--   psql -U postgres -f docker/init_db.sql
--
-- Usage (pgAdmin / DBeaver):
--   Open this file and execute.
-- ============================================================================

-- ── 1. Create the database ──────────────────────────────────────────────────
-- Connect as a superuser (e.g. postgres) to run this section.

-- Create the application user (skip if it already exists)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'llm_user') THEN
        CREATE ROLE llm_user WITH LOGIN PASSWORD 'llm_pass';
    END IF;
END
$$;

-- Create the database (must be run outside a transaction in psql)
-- If running in psql you can uncomment the next line instead:
--   CREATE DATABASE llm_logs OWNER llm_user;

SELECT 'CREATE DATABASE llm_logs OWNER llm_user'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'llm_logs')
\gexec

-- ── 2. Connect to the new database ─────────────────────────────────────────
\connect llm_logs

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE llm_logs TO llm_user;

-- ── 3. Enable required extensions ──────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── 4. Create tables ───────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS request_logs (
    -- Primary key: UUID v4
    id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Task metadata
    task_type       VARCHAR(32)     NOT NULL,
    input_text      TEXT            NOT NULL,
    output_text     TEXT,

    -- Lifecycle
    status          VARCHAR(16)     NOT NULL DEFAULT 'pending',
    tokens_used     INTEGER,
    latency_ms      DOUBLE PRECISION,
    error_message   TEXT,

    -- Timestamps
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ
);

-- ── 5. Indexes ─────────────────────────────────────────────────────────────
-- These match the index=True columns in the SQLAlchemy model.

CREATE INDEX IF NOT EXISTS ix_request_logs_task_type
    ON request_logs (task_type);

CREATE INDEX IF NOT EXISTS ix_request_logs_status
    ON request_logs (status);

-- Optional: speed up "most recent requests" queries
CREATE INDEX IF NOT EXISTS ix_request_logs_created_at
    ON request_logs (created_at DESC);

-- ── 6. Grant table-level privileges to the app user ────────────────────────
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO llm_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO llm_user;

-- Make future tables accessible too
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL PRIVILEGES ON TABLES TO llm_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO llm_user;

-- ============================================================================
-- Done! Your database is ready for MC-LLM.
-- ============================================================================

-- ── TTS Logging ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS tts_requests (
    id             SERIAL PRIMARY KEY,
    text           TEXT NOT NULL,
    language       VARCHAR(10) NOT NULL DEFAULT 'ar',
    speed          FLOAT NOT NULL DEFAULT 1.0,
    format         VARCHAR(10) NOT NULL DEFAULT 'wav',
    has_ref_audio  BOOLEAN NOT NULL DEFAULT FALSE,
    duration_ms    FLOAT,
    audio_size_b   INTEGER,
    status         VARCHAR(20) NOT NULL DEFAULT 'success',
    error_message  TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tts_requests_created_at
    ON tts_requests (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_tts_requests_status
    ON tts_requests (status);

-- Grant access to the app user
GRANT ALL PRIVILEGES ON TABLE tts_requests TO llm_user;
GRANT USAGE, SELECT ON SEQUENCE tts_requests_id_seq TO llm_user;
