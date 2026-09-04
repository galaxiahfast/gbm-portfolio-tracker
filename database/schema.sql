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
    created_at TEXT NOT NULL, integrity_version INTEGER NOT NULL DEFAULT 1, available_at TEXT, source_bar_at TEXT, horizon_policy TEXT, resolution_status TEXT NOT NULL DEFAULT 'LEGACY_UNVERIFIED', outcome_bar_at TEXT, outcome_source TEXT, resolution_sha256 TEXT,
    UNIQUE(symbol, observed_at, horizon_minutes)
);

CREATE TABLE operational_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL, payload_json TEXT NOT NULL,
            previous_sha256 TEXT NOT NULL, sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL
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

CREATE TABLE zone_daily_validation (
            sha256 TEXT PRIMARY KEY, session_date TEXT NOT NULL,
            symbol TEXT NOT NULL, model_version_hash TEXT NOT NULL,
            created_at TEXT NOT NULL, payload_json TEXT NOT NULL);

CREATE TABLE zone_market_evidence (
            sha256 TEXT PRIMARY KEY, payload_json TEXT NOT NULL);

CREATE TABLE zone_prediction_log (
            id TEXT PRIMARY KEY, symbol TEXT NOT NULL,
            timestamp_prediction TEXT NOT NULL, source_bar_closed_at TEXT NOT NULL,
            session_date TEXT NOT NULL, expires_at TEXT NOT NULL,
            zone_key TEXT NOT NULL, zone_type TEXT NOT NULL,
            zone_price REAL NOT NULL, zone_low REAL NOT NULL, zone_high REAL NOT NULL,
            reference_price REAL NOT NULL,
            predicted_touch_probability REAL NOT NULL CHECK(predicted_touch_probability BETWEEN 0 AND 1),
            predicted_close_above_probability REAL,
            predicted_close_probability REAL, close_direction TEXT NOT NULL,
            timeframe TEXT NOT NULL, model_version_hash TEXT NOT NULL, model_name TEXT NOT NULL,
            context_json TEXT NOT NULL, touch_eligible INTEGER NOT NULL,
            forecast_sha256 TEXT NOT NULL,
            actual_touch_occurred INTEGER, actual_close_price REAL,
            actual_close_relation TEXT, resolved_at TEXT, resolution_note TEXT,
            evidence_sha256 TEXT, resolution_sha256 TEXT,
            UNIQUE(symbol, model_version_hash, source_bar_closed_at, zone_key)
        );

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

CREATE INDEX idx_live_pending_maturity
            ON live_model_observations(symbol, resolution_status, available_at);

CREATE INDEX idx_operational_symbol ON operational_events(symbol, id);

CREATE INDEX idx_portfolio_snapshots_time
    ON portfolio_snapshots(observed_at DESC);

CREATE INDEX idx_prices_symbol_time
    ON price_snapshots(symbol, observed_at DESC);

CREATE INDEX idx_trades_symbol_time
    ON trades(symbol, executed_at, id);

CREATE INDEX ix_zone_pending ON zone_prediction_log(resolved_at, expires_at);

CREATE TRIGGER live_forecast_immutable
            BEFORE UPDATE ON live_model_observations
            WHEN OLD.integrity_version = 2 AND (NEW.symbol IS NOT OLD.symbol OR NEW.observed_at IS NOT OLD.observed_at OR NEW.available_at IS NOT OLD.available_at OR NEW.horizon_minutes IS NOT OLD.horizon_minutes OR NEW.reference_price IS NOT OLD.reference_price OR NEW.raw_probability_up IS NOT OLD.raw_probability_up OR NEW.predicted_direction IS NOT OLD.predicted_direction OR NEW.parameters_json IS NOT OLD.parameters_json OR NEW.source_bar_at IS NOT OLD.source_bar_at OR NEW.horizon_policy IS NOT OLD.horizon_policy OR NEW.integrity_version IS NOT OLD.integrity_version OR NEW.created_at IS NOT OLD.created_at OR NEW.observation_sha256 IS NOT OLD.observation_sha256 OR NEW.id IS NOT OLD.id)
            BEGIN SELECT RAISE(ABORT, 'live_forecast_immutable'); END;

CREATE TRIGGER live_observation_no_delete
            BEFORE DELETE ON live_model_observations WHEN OLD.integrity_version = 2
            BEGIN SELECT RAISE(ABORT, 'live_observation_immutable'); END;

CREATE TRIGGER live_resolution_immutable
            BEFORE UPDATE ON live_model_observations
            WHEN OLD.integrity_version = 2 AND OLD.resolution_status != 'PENDING'
            BEGIN SELECT RAISE(ABORT, 'live_resolution_immutable'); END;

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

CREATE TRIGGER zone_daily_validation_no_delete BEFORE DELETE ON zone_daily_validation
                BEGIN SELECT RAISE(ABORT, 'append-only evidence'); END;

CREATE TRIGGER zone_daily_validation_no_update BEFORE UPDATE ON zone_daily_validation
                    BEGIN SELECT RAISE(ABORT, 'immutable evidence'); END;

CREATE TRIGGER zone_forecast_immutable
            BEFORE UPDATE ON zone_prediction_log WHEN NEW.id IS NOT OLD.id OR NEW.symbol IS NOT OLD.symbol OR NEW.timestamp_prediction IS NOT OLD.timestamp_prediction OR NEW.source_bar_closed_at IS NOT OLD.source_bar_closed_at OR NEW.session_date IS NOT OLD.session_date OR NEW.expires_at IS NOT OLD.expires_at OR NEW.zone_key IS NOT OLD.zone_key OR NEW.zone_type IS NOT OLD.zone_type OR NEW.zone_price IS NOT OLD.zone_price OR NEW.zone_low IS NOT OLD.zone_low OR NEW.zone_high IS NOT OLD.zone_high OR NEW.reference_price IS NOT OLD.reference_price OR NEW.predicted_touch_probability IS NOT OLD.predicted_touch_probability OR NEW.predicted_close_above_probability IS NOT OLD.predicted_close_above_probability OR NEW.predicted_close_probability IS NOT OLD.predicted_close_probability OR NEW.close_direction IS NOT OLD.close_direction OR NEW.timeframe IS NOT OLD.timeframe OR NEW.model_version_hash IS NOT OLD.model_version_hash OR NEW.model_name IS NOT OLD.model_name OR NEW.context_json IS NOT OLD.context_json OR NEW.touch_eligible IS NOT OLD.touch_eligible OR NEW.forecast_sha256 IS NOT OLD.forecast_sha256
            BEGIN SELECT RAISE(ABORT, 'immutable zone forecast'); END;

CREATE TRIGGER zone_market_evidence_no_delete BEFORE DELETE ON zone_market_evidence
                BEGIN SELECT RAISE(ABORT, 'append-only evidence'); END;

CREATE TRIGGER zone_market_evidence_no_update BEFORE UPDATE ON zone_market_evidence
                    BEGIN SELECT RAISE(ABORT, 'immutable evidence'); END;

CREATE TRIGGER zone_prediction_log_no_delete BEFORE DELETE ON zone_prediction_log
                BEGIN SELECT RAISE(ABORT, 'append-only evidence'); END;

CREATE TRIGGER zone_resolution_immutable
            BEFORE UPDATE ON zone_prediction_log WHEN OLD.resolved_at IS NOT NULL
            BEGIN SELECT RAISE(ABORT, 'immutable zone resolution'); END;
