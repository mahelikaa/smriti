CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE memories
    ADD COLUMN embedding vector(1024),
    ADD COLUMN embedding_model text,
    ADD CONSTRAINT memories_embedding_pair_chk
        CHECK ((embedding IS NULL) = (embedding_model IS NULL));