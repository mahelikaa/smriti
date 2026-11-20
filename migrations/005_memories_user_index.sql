CREATE INDEX CONCURRENTLY IF NOT EXISTS memories_user_model_idx
    ON memories (user_id, embedding_model);