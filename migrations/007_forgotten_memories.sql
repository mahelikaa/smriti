CREATE TABLE forgotten_memories (
    memory_id bigint PRIMARY KEY,
    user_id text NOT NULL,
    forgotten_at timestamptz NOT NULL DEFAULT now()
);