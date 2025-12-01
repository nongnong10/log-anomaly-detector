-- Table: event
CREATE TABLE event
(
    event_id       VARCHAR PRIMARY KEY,
    event_template TEXT,
    occurrences    INT DEFAULT 0
);

-- Table: log_block
CREATE TABLE log_block
(
    block_id       VARCHAR PRIMARY KEY,
    event_sequence VARCHAR[],
    has_data BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    anomaly_score DOUBLE PRECISION
);


-- Table: log_line
CREATE TABLE log_line
(
    line_id   SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    pid       INT,
    level     VARCHAR, -- INFO / WARN / ERROR (no enum constraint)
    component VARCHAR,
    content   TEXT,
    event_id  VARCHAR, -- no FK
    block_id  VARCHAR  -- no FK
);

-- Table: anomaly_sequence
CREATE TABLE anomaly_sequence
(
    block_id VARCHAR PRIMARY KEY,
    label    VARCHAR -- 'Normal' / 'Anomaly'
);

-- Table: event_mapping
CREATE TABLE event_mapping
(
    event_id            VARCHAR PRIMARY KEY,
    event_mapping_id    INT
);