-- Esquema generado automáticamente; no contiene datos.
CREATE TABLE analytics_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    as_of TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value TEXT NOT NULL,
    source TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(symbol, as_of, feature_name, source)
);

CREATE TABLE audit_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL CHECK (status IN ('PASS', 'WARNING', 'ERROR')),
                passed INTEGER NOT NULL,
                warnings INTEGER NOT NULL,
                errors INTEGER NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

CREATE TABLE backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    engine_version TEXT NOT NULL,
    symbols_json TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    dataset_sha256 TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('APPROVED', 'REJECTED')),
    created_at TEXT NOT NULL
);

CREATE TABLE cash_movements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('INITIAL', 'DEPOSIT', 'WITHDRAWAL')),
    original_amount TEXT NOT NULL,
    original_currency TEXT NOT NULL CHECK (original_currency IN ('USD', 'MXN')),
    usd_amount TEXT NOT NULL,
    fx_rate TEXT,
    occurred_at TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE fundamental_news_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    provider TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE fx_rates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    base_currency TEXT NOT NULL,
    quote_currency TEXT NOT NULL,
    rate TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    provider TEXT NOT NULL,
    is_reference INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE live_model_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    reference_price TEXT NOT NULL,
    raw_probability_up TEXT NOT NULL,
    predicted_direction TEXT NOT NULL CHECK (predicted_direction IN ('UP', 'DOWN')),
    parameters_json TEXT NOT NULL,
    observation_sha256 TEXT NOT NULL,
    outcome_price TEXT,
    outcome_up INTEGER,
    successful INTEGER,
    resolved_at TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(symbol, observed_at, horizon_minutes)
);

CREATE TABLE portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cash_usd TEXT NOT NULL,
    holdings_value_usd TEXT NOT NULL,
    equity_usd TEXT NOT NULL,
    fx_rate TEXT NOT NULL,
    observed_at TEXT NOT NULL
);

CREATE TABLE prediction_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    parameters_json TEXT NOT NULL DEFAULT '{}',
    training_cutoff TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES prediction_runs(id),
    symbol TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    probability_up TEXT NOT NULL,
    expected_return TEXT,
    confidence TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE price_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    price_usd TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    provider TEXT NOT NULL,
    is_manual INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256 TEXT NOT NULL UNIQUE,
    original_filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    original_path TEXT NOT NULL,
    thumbnail_path TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    byte_size INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                );

CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    product TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    order_type TEXT NOT NULL,
    quantity TEXT NOT NULL,
    price_usd TEXT NOT NULL,
    commission_usd TEXT NOT NULL,
    commission_rate_pct TEXT,
    gross_usd TEXT NOT NULL,
    reported_total_usd TEXT,
    cash_delta_usd TEXT NOT NULL,
    fx_rate TEXT,
    executed_at TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    receipt_id INTEGER REFERENCES receipts(id),
    ocr_text TEXT NOT NULL DEFAULT '',
    ocr_confidence TEXT,
    validation_status TEXT NOT NULL DEFAULT 'PENDING',
    created_at TEXT NOT NULL
, reported_total_type TEXT
                NOT NULL DEFAULT 'GROSS'
                CHECK (reported_total_type IN ('GROSS', 'SETTLEMENT')));

CREATE INDEX idx_audit_runs_time
            ON audit_runs(created_at DESC)
            ;

CREATE INDEX idx_backtest_runs_time
    ON backtest_runs(created_at DESC, id DESC);

CREATE INDEX idx_cash_time
    ON cash_movements(occurred_at, id);

CREATE INDEX idx_fundamental_news_symbol_time
    ON fundamental_news_snapshots(symbol, observed_at DESC, id DESC);

CREATE INDEX idx_fx_pair_time
    ON fx_rates(base_currency, quote_currency, observed_at DESC);

CREATE INDEX idx_live_model_symbol_time
    ON live_model_observations(symbol, observed_at DESC);

CREATE INDEX idx_portfolio_snapshots_time
    ON portfolio_snapshots(observed_at DESC);

CREATE INDEX idx_prices_symbol_time
    ON price_snapshots(symbol, observed_at DESC);

CREATE INDEX idx_trades_symbol_time
    ON trades(symbol, executed_at, id);

CREATE TRIGGER prevent_duplicate_trade_receipt_insert
            BEFORE INSERT ON trades
            WHEN NEW.receipt_id IS NOT NULL AND EXISTS (
                SELECT 1 FROM trades WHERE receipt_id = NEW.receipt_id
            )
            BEGIN
                SELECT RAISE(ABORT, 'receipt_already_linked');
            END;

CREATE TRIGGER prevent_duplicate_trade_receipt_update
            BEFORE UPDATE OF receipt_id ON trades
            WHEN NEW.receipt_id IS NOT NULL AND EXISTS (
                SELECT 1 FROM trades
                WHERE receipt_id = NEW.receipt_id AND id <> OLD.id
            )
            BEGIN
                SELECT RAISE(ABORT, 'receipt_already_linked');
            END;
