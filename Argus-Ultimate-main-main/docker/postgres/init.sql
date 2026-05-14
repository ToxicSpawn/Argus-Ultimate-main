-- Push 72 — Argus Postgres schema initialisation
-- Tables: trades, positions, equity_snapshots, alert_events,
--         backtest_results, order_log

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -----------------------------------------------------------------------
-- Trade history
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trades (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol          VARCHAR(20) NOT NULL,
    side            VARCHAR(10) NOT NULL,          -- long | short
    qty             NUMERIC(20,8) NOT NULL,
    entry_price     NUMERIC(20,8) NOT NULL,
    exit_price      NUMERIC(20,8),
    commission      NUMERIC(20,8) DEFAULT 0,
    pnl             NUMERIC(20,8),
    strategy        VARCHAR(50),
    algorithm       VARCHAR(20),                   -- PPO | TD3 | SAC
    mode            VARCHAR(10) DEFAULT 'paper',   -- paper | live
    opened_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at       TIMESTAMPTZ,
    metadata        JSONB
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol    ON trades (symbol);
CREATE INDEX IF NOT EXISTS idx_trades_opened_at ON trades (opened_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_strategy  ON trades (strategy);

-- -----------------------------------------------------------------------
-- Live positions (current snapshot)
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS positions (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol          VARCHAR(20) NOT NULL UNIQUE,
    side            VARCHAR(10) NOT NULL,
    qty             NUMERIC(20,8) NOT NULL,
    avg_entry       NUMERIC(20,8) NOT NULL,
    unrealised_pnl  NUMERIC(20,8) DEFAULT 0,
    leverage        NUMERIC(6,2)  DEFAULT 1,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -----------------------------------------------------------------------
-- Equity curve snapshots
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id              BIGSERIAL   PRIMARY KEY,
    equity_usd      NUMERIC(20,4) NOT NULL,
    cash_usd        NUMERIC(20,4) NOT NULL,
    daily_pnl       NUMERIC(20,4) DEFAULT 0,
    drawdown_pct    NUMERIC(8,4)  DEFAULT 0,
    n_positions     INTEGER       DEFAULT 0,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_equity_recorded_at ON equity_snapshots (recorded_at DESC);

-- -----------------------------------------------------------------------
-- Alert event log
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alert_events (
    id              BIGSERIAL   PRIMARY KEY,
    rule_name       VARCHAR(80) NOT NULL,
    severity        VARCHAR(20) NOT NULL,
    message         TEXT,
    value           NUMERIC(20,8),
    threshold       NUMERIC(20,8),
    dispatched      BOOLEAN DEFAULT FALSE,
    fired_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -----------------------------------------------------------------------
-- Backtest results summary
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtest_results (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy        VARCHAR(50) NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10),
    start_date      DATE,
    end_date        DATE,
    sharpe          NUMERIC(10,4),
    sortino         NUMERIC(10,4),
    calmar          NUMERIC(10,4),
    max_drawdown_pct NUMERIC(8,4),
    cagr_pct        NUMERIC(10,4),
    total_return_pct NUMERIC(10,4),
    n_trades        INTEGER,
    win_rate        NUMERIC(6,4),
    mc_ruin_prob    NUMERIC(6,4),
    mc_median_sharpe NUMERIC(10,4),
    wf_mean_sharpe  NUMERIC(10,4),
    full_report     JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -----------------------------------------------------------------------
-- Order log
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS order_log (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    local_id        VARCHAR(20),
    exchange_id     VARCHAR(50),
    symbol          VARCHAR(20) NOT NULL,
    side            VARCHAR(10) NOT NULL,
    order_type      VARCHAR(10) NOT NULL,
    qty             NUMERIC(20,8),
    price           NUMERIC(20,8),
    filled_qty      NUMERIC(20,8) DEFAULT 0,
    avg_fill_price  NUMERIC(20,8),
    commission      NUMERIC(20,8) DEFAULT 0,
    status          VARCHAR(20),
    mode            VARCHAR(10) DEFAULT 'paper',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_order_log_symbol     ON order_log (symbol);
CREATE INDEX IF NOT EXISTS idx_order_log_created_at ON order_log (created_at DESC);
