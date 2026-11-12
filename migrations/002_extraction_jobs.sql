CREATE TABLE extraction_jobs (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    turn_id bigint NOT NULL UNIQUE
    REFERENCES turns(id) ON DELETE CASCADE,

    status text NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'running', 'done', 'dead')),

    attempts int NOT NULL DEFAULT 0,

    run_after timestamptz NOT NULL DEFAULT now(),

    locked_until timestamptz,

    last_error text,

    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX  extraction_jobs_pending_idx
ON extraction_jobs (run_after, id)
WHERE status = 'pending';