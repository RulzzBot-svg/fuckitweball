# NBA Bets - Data Clean Room

Phase 1 initializes a local SQLite database and ingests historical NBA games from BALLDONTLIE.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Create your local env file:
   - Copy `.env.example` to `.env`
   - Set `BALLDONTLIE_API_KEY`

## Run ingestion

- Pull both seasons (recommended):
  - `python ingest_history.py --seasons 2024 2025`
- Pull one season:
  - `python ingest_history.py --seasons 2024`

The script creates `nba_bets.db` with:
- `teams` table (metadata)
- `games` table (historical finished games)

## Verify counts

- `python db_summary.py`

## Phase 2: Feature engineering

- Build team-level training features from historical games:
  - `python build_features.py`
- Build features for specific seasons only:
  - `python build_features.py --seasons 2024 2025`

This creates `team_game_features` with one row per team per finished game, including:
- Pregame win/loss streaks
- Point margin
- Pregame rest days

## Phase 3: Predictor + virtual bankroll

- Train on 2024 and backtest on 2025:
  - `python train_backtest.py`
- Override seasons and bankroll settings:
  - `python train_backtest.py --train-season 2024 --test-season 2025 --start-bankroll 1000 --stake 25 --min-confidence 0.55 --odds -110`

This writes model output to:
- `model_backtest_runs` (one row per experiment)
- `model_game_predictions` (one row per test game)

## Notes

- Uses cursor-based pagination via `meta.next_cursor`.
- Handles `429` with retries and backoff.
- Stores only finished games (`status == Final`).
- Uses `v1/games` for historical game ingestion.
