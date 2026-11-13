CREATE TABLE memories (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id text NOT NULL,
    turn_id bigint NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    kind text NOT NULL CHECK (kind IN ('fact', 'preference', 'event', 'relationship')),
    content text NOT NULL,
    importance smallint NOT NULL CHECK (importance BETWEEN 1 AND 5),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX memories_turn_id_idx ON memories (turn_id);