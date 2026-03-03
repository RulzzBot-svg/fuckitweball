import argparse
import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier
import joblib
from pathlib import Path

load_dotenv()

DEFAULT_DB_NAME = os.getenv("DB_NAME", "nba_bets.db")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost on one season and backtest on another with virtual bankroll"
    )
    parser.add_argument("--db", default=DEFAULT_DB_NAME, help="SQLite DB path")
    parser.add_argument("--train-season", type=int, default=2024)
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--start-bankroll", type=float, default=1000.0)
    parser.add_argument("--stake", type=float, default=25.0)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.55,
        help="Only place bets where max(P(home), P(away)) >= threshold",
    )
    parser.add_argument(
        "--odds",
        type=int,
        default=-110,
        help="American odds used for bankroll simulation (e.g. -110)",
    )
    return parser.parse_args()


def init_model_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS model_backtest_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            model_name TEXT NOT NULL,
            train_season INTEGER NOT NULL,
            test_season INTEGER NOT NULL,
            train_games INTEGER NOT NULL,
            test_games INTEGER NOT NULL,
            test_accuracy REAL NOT NULL,
            test_log_loss REAL NOT NULL,
            test_roc_auc REAL NOT NULL,
            bets_placed INTEGER NOT NULL,
            bet_win_rate REAL,
            start_bankroll REAL NOT NULL,
            final_bankroll REAL NOT NULL,
            roi_pct REAL NOT NULL,
            min_confidence REAL NOT NULL,
            stake REAL NOT NULL,
            odds INTEGER NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS model_game_predictions (
            run_id TEXT NOT NULL,
            game_id INTEGER NOT NULL,
            season INTEGER NOT NULL,
            game_date TEXT NOT NULL,
            home_team_id INTEGER NOT NULL,
            visitor_team_id INTEGER NOT NULL,
            home_win INTEGER NOT NULL,
            prob_home_win REAL NOT NULL,
            prob_away_win REAL NOT NULL,
            pick_side TEXT NOT NULL,
            confidence REAL NOT NULL,
            bet_placed INTEGER NOT NULL,
            bet_amount REAL,
            odds INTEGER,
            won_bet INTEGER,
            bankroll_after REAL,
            PRIMARY KEY (run_id, game_id)
        )
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_model_predictions_season
        ON model_game_predictions (season)
        """
    )

    conn.commit()


def load_model_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
        SELECT
            g.id AS game_id,
            g.date AS game_date,
            g.season,
            g.home_team_id,
            g.visitor_team_id,
            g.home_score,
            g.visitor_score,
            h.pregame_win_streak AS home_pregame_win_streak,
            h.pregame_loss_streak AS home_pregame_loss_streak,
            h.pregame_rest_days AS home_pregame_rest_days,
            a.pregame_win_streak AS away_pregame_win_streak,
            a.pregame_loss_streak AS away_pregame_loss_streak,
            a.pregame_rest_days AS away_pregame_rest_days
        FROM games g
        INNER JOIN team_game_features h
            ON g.id = h.game_id
           AND h.team_id = g.home_team_id
        INNER JOIN team_game_features a
            ON g.id = a.game_id
           AND a.team_id = g.visitor_team_id
        WHERE g.home_score IS NOT NULL
          AND g.visitor_score IS NOT NULL
        ORDER BY g.date, g.id
    """

    df = pd.read_sql_query(query, conn)
    if df.empty:
        raise ValueError("No games found. Run ingestion + feature build first.")

    df["home_win"] = (df["home_score"] > df["visitor_score"]).astype(int)

    for col in ["home_pregame_rest_days", "away_pregame_rest_days"]:
        df[col] = df[col].fillna(2.0)

    df["streak_net_home"] = df["home_pregame_win_streak"] - df["home_pregame_loss_streak"]
    df["streak_net_away"] = df["away_pregame_win_streak"] - df["away_pregame_loss_streak"]
    df["streak_net_diff"] = df["streak_net_home"] - df["streak_net_away"]
    df["rest_days_diff"] = df["home_pregame_rest_days"] - df["away_pregame_rest_days"]

    return df


def get_feature_columns() -> list[str]:
    return [
        "home_pregame_win_streak",
        "home_pregame_loss_streak",
        "home_pregame_rest_days",
        "away_pregame_win_streak",
        "away_pregame_loss_streak",
        "away_pregame_rest_days",
        "streak_net_home",
        "streak_net_away",
        "streak_net_diff",
        "rest_days_diff",
    ]


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1 + (odds / 100)
    return 1 + (100 / abs(odds))


def run_backtest(
    df_test: pd.DataFrame,
    prob_home: pd.Series,
    start_bankroll: float,
    stake: float,
    min_confidence: float,
    odds: int,
) -> tuple[pd.DataFrame, float, int, float]:
    bankroll = start_bankroll
    decimal_odds = american_to_decimal(odds)

    rows = []
    bets_placed = 0
    bets_won = 0

    for row, p_home in zip(df_test.itertuples(index=False), prob_home):
        p_away = 1.0 - float(p_home)
        pick_side = "home" if p_home >= 0.5 else "away"
        confidence = max(float(p_home), p_away)

        bet_placed = int(confidence >= min_confidence)
        won_bet = None
        bet_amount = None

        if bet_placed:
            bets_placed += 1
            bet_amount = stake

            is_correct = (pick_side == "home" and row.home_win == 1) or (
                pick_side == "away" and row.home_win == 0
            )
            won_bet = int(is_correct)

            if is_correct:
                bets_won += 1
                bankroll += stake * (decimal_odds - 1)
            else:
                bankroll -= stake

        rows.append(
            {
                "game_id": row.game_id,
                "season": row.season,
                "game_date": row.game_date,
                "home_team_id": row.home_team_id,
                "visitor_team_id": row.visitor_team_id,
                "home_win": int(row.home_win),
                "prob_home_win": float(p_home),
                "prob_away_win": p_away,
                "pick_side": pick_side,
                "confidence": confidence,
                "bet_placed": bet_placed,
                "bet_amount": bet_amount,
                "odds": odds if bet_placed else None,
                "won_bet": won_bet,
                "bankroll_after": bankroll,
            }
        )

    bet_win_rate = (bets_won / bets_placed) if bets_placed else 0.0
    return pd.DataFrame(rows), bankroll, bets_placed, bet_win_rate


def save_results(
    conn: sqlite3.Connection,
    run_id: str,
    created_at: str,
    args: argparse.Namespace,
    train_games: int,
    test_games: int,
    accuracy: float,
    ll: float,
    roc_auc: float,
    predictions_df: pd.DataFrame,
    final_bankroll: float,
    bets_placed: int,
    bet_win_rate: float,
) -> None:
    cur = conn.cursor()

    roi_pct = ((final_bankroll - args.start_bankroll) / args.start_bankroll) * 100.0

    cur.execute(
        """
        INSERT INTO model_backtest_runs (
            run_id, created_at, model_name, train_season, test_season,
            train_games, test_games, test_accuracy, test_log_loss, test_roc_auc,
            bets_placed, bet_win_rate, start_bankroll, final_bankroll, roi_pct,
            min_confidence, stake, odds
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            created_at,
            "xgboost",
            args.train_season,
            args.test_season,
            train_games,
            test_games,
            accuracy,
            ll,
            roc_auc,
            bets_placed,
            bet_win_rate,
            args.start_bankroll,
            final_bankroll,
            roi_pct,
            args.min_confidence,
            args.stake,
            args.odds,
        ),
    )

    to_insert = predictions_df.copy()
    to_insert.insert(0, "run_id", run_id)

    cur.executemany(
        """
        INSERT INTO model_game_predictions (
            run_id, game_id, season, game_date, home_team_id, visitor_team_id,
            home_win, prob_home_win, prob_away_win, pick_side, confidence,
            bet_placed, bet_amount, odds, won_bet, bankroll_after
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [tuple(row) for row in to_insert.itertuples(index=False, name=None)],
    )

    conn.commit()


def main() -> None:
    args = parse_args()

    conn = sqlite3.connect(args.db)
    try:
        init_model_tables(conn)
        df = load_model_frame(conn)

        df_train = df[df["season"] == args.train_season].copy()
        df_test = df[df["season"] == args.test_season].copy()

        if df_train.empty:
            raise ValueError(f"No games found for train season {args.train_season}.")
        if df_test.empty:
            raise ValueError(f"No games found for test season {args.test_season}.")

        features = get_feature_columns()
        x_train = df_train[features]
        y_train = df_train["home_win"]
        x_test = df_test[features]
        y_test = df_test["home_win"]

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=350,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=4,
        )
        model.fit(x_train, y_train)

        # Persist trained model for Phase 4 UI
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f"xgb_{args.train_season}_to_{args.test_season}.pkl"
        try:
            joblib.dump(model, model_path)
            print(f"Saved trained model to {model_path}")
        except Exception:
            # best-effort save; continue even if persistence fails
            print("Warning: failed to save trained model artifact")

        prob_home = model.predict_proba(x_test)[:, 1]
        preds = (prob_home >= 0.5).astype(int)

        accuracy = float(accuracy_score(y_test, preds))
        ll = float(log_loss(y_test, prob_home, labels=[0, 1]))
        roc_auc = float(roc_auc_score(y_test, prob_home))

        predictions_df, final_bankroll, bets_placed, bet_win_rate = run_backtest(
            df_test=df_test,
            prob_home=pd.Series(prob_home),
            start_bankroll=args.start_bankroll,
            stake=args.stake,
            min_confidence=args.min_confidence,
            odds=args.odds,
        )

        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
        created_at = datetime.now(timezone.utc).isoformat()

        save_results(
            conn=conn,
            run_id=run_id,
            created_at=created_at,
            args=args,
            train_games=len(df_train),
            test_games=len(df_test),
            accuracy=accuracy,
            ll=ll,
            roc_auc=roc_auc,
            predictions_df=predictions_df,
            final_bankroll=final_bankroll,
            bets_placed=bets_placed,
            bet_win_rate=bet_win_rate,
        )

        roi_pct = ((final_bankroll - args.start_bankroll) / args.start_bankroll) * 100.0

        print("Phase 3 training + backtest complete")
        print(f"Run ID: {run_id}")
        print(f"Train season: {args.train_season} | games: {len(df_train)}")
        print(f"Test season: {args.test_season} | games: {len(df_test)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Log loss: {ll:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Bets placed: {bets_placed}")
        print(f"Bet win rate: {bet_win_rate:.4f}")
        print(f"Start bankroll: {args.start_bankroll:.2f}")
        print(f"Final bankroll: {final_bankroll:.2f}")
        print(f"ROI %: {roi_pct:.2f}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
