import sqlite3
from pathlib import Path
import pandas as pd
import joblib
import datetime
import sys

DB = Path("nba_bets.db")
MODELS_DIR = Path("models")

FEATURES = [
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


def find_latest_model():
    if not MODELS_DIR.exists():
        return None
    models = sorted(MODELS_DIR.glob("*.pkl"), reverse=True)
    return models[0] if models else None


def load_games_for_date(conn, date_str):
    q = """
    SELECT
        g.id AS game_id,
        g.date AS game_date,
        g.season,
        g.home_team_id,
        g.visitor_team_id,
        h.pregame_win_streak AS home_pregame_win_streak,
        h.pregame_loss_streak AS home_pregame_loss_streak,
        h.pregame_rest_days AS home_pregame_rest_days,
        a.pregame_win_streak AS away_pregame_win_streak,
        a.pregame_loss_streak AS away_pregame_loss_streak,
        a.pregame_rest_days AS away_pregame_rest_days
    FROM games g
    JOIN team_game_features h ON g.id = h.game_id AND h.team_id = g.home_team_id
    JOIN team_game_features a ON g.id = a.game_id AND a.team_id = g.visitor_team_id
    WHERE g.home_score IS NULL
      AND g.date LIKE ?
    ORDER BY g.date
    """
    return pd.read_sql_query(q, conn, params=(f"{date_str}%",))


def main():
    if not DB.exists():
        print("Database not found: nba_bets.db")
        raise SystemExit(1)

    model_file = find_latest_model()
    if not model_file:
        print("No model artifact found in models/ — run training first.")
        raise SystemExit(1)

    print(f"Using model: {model_file}")
    model = joblib.load(model_file)

    today = datetime.date.today().isoformat()
    conn = sqlite3.connect(str(DB))
    df = load_games_for_date(conn, today)
    conn.close()

    if df.empty:
        print(f"No upcoming games with features found for date {today}.")
        print("If you want to predict a specific game, ensure it exists in `games` and has `team_game_features` entries.")
        raise SystemExit(0)

    # derived features
    df["streak_net_home"] = df["home_pregame_win_streak"] - df["home_pregame_loss_streak"]
    df["streak_net_away"] = df["away_pregame_win_streak"] - df["away_pregame_loss_streak"]
    df["streak_net_diff"] = df["streak_net_home"] - df["streak_net_away"]
    df["rest_days_diff"] = df["home_pregame_rest_days"] - df["away_pregame_rest_days"]

    X = df[FEATURES]
    probs = model.predict_proba(X)[:, 1]

    out_rows = []
    for r, p_home in zip(df.itertuples(index=False), probs):
        p_away = 1.0 - float(p_home)
        pick = "home" if p_home >= 0.5 else "away"
        confidence = max(float(p_home), p_away)
        out_rows.append(
            {
                "game_id": r.game_id,
                "game_date": r.game_date,
                "home_team_id": r.home_team_id,
                "visitor_team_id": r.visitor_team_id,
                "prob_home_win": float(p_home),
                "prob_away_win": float(p_away),
                "pick_side": pick,
                "confidence": confidence,
            }
        )

    out_df = pd.DataFrame(out_rows)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("Predictions for date:", today)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
