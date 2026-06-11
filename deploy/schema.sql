-- SpendScope backend — MySQL schema (cache + rate limiting + operational logging).
-- Holds NO user data: no transactions, no statements, no receipts (SPECS §11.4).
-- The app also creates these on startup (db.init_schema); this file is for manual provisioning.

CREATE DATABASE IF NOT EXISTS spendscope CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE spendscope;

-- Categorization cache: identical concept -> reuse category, skip the LLM (SPECS §4.2).
CREATE TABLE IF NOT EXISTS categorize_cache (
    concept_key   VARCHAR(255) NOT NULL,
    category      VARCHAR(255) NOT NULL,
    confidence    FLOAT NULL,
    updated_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                  ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (concept_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Per-user fixed-window rate limiting (SPECS §11.4).
CREATE TABLE IF NOT EXISTS rate_limit (
    user_id       VARCHAR(255) NOT NULL,
    endpoint      VARCHAR(64) NOT NULL,
    window_start  BIGINT NOT NULL,
    hits          INT NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, endpoint)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Operational request metrics (SPECS §11.5). No user data — provider/latency/status only.
CREATE TABLE IF NOT EXISTS request_log (
    id             BIGINT AUTO_INCREMENT PRIMARY KEY,
    ts             TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    endpoint       VARCHAR(32) NOT NULL,
    provider_used  VARCHAR(32) NULL,
    is_fallback    TINYINT(1) NOT NULL DEFAULT 0,
    primary_error  VARCHAR(255) NULL,
    latency_ms     INT NULL,
    status         INT NULL,
    INDEX idx_ts (ts),
    INDEX idx_endpoint (endpoint)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
