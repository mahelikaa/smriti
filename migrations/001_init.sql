CREATE TABLE turns (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id text NOT NULL,
    external_id text NOT NULL,
    role text NOT NULL CHECK(role IN ('user', 'assistant')),
    text text NOT NULL,
    sent_at timestamptz NOT NULL,
    UNIQUE(user_id, external_id)
);

CREATE INDEX turns_user_sent_idx ON turns (user_id, sent_at, id);