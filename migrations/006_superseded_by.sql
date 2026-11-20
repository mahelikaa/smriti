ALTER TABLE memories
    ADD COLUMN superseded_by bigint REFERENCES memories(id) ON DELETE SET NULL;
