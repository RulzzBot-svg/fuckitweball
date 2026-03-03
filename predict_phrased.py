import sqlite3
from pathlib import Path
import pandas as pd
import joblib
import datetime

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
        ht.full_name AS home_team_name,
        vt.full_name AS visitor_team_name,
        h.pregame_win_streak AS home_pregame_win_streak,
        h.pregame_loss_streak AS home_pregame_loss_streak,
        h.pregame_rest_days AS home_pregame_rest_days,
        a.pregame_win_streak AS away_pregame_win_streak,
        a.pregame_loss_streak AS away_pregame_loss_streak,
        a.pregame_rest_days AS away_pregame_rest_days
    FROM games g
    JOIN teams ht ON g.home_team_id = ht.id
    JOIN teams vt ON g.visitor_team_id = vt.id
    JOIN team_game_features h ON g.id = h.game_id AND h.team_id = g.home_team_id
    JOIN team_game_features a ON g.id = a.game_id AND a.team_id = g.visitor_team_id
    WHERE g.home_score IS NULL
      AND g.date LIKE ?
    ORDER BY g.date
    """
    return pd.read_sql_query(q, conn, params=(f"{date_str}%",))


def label_from_prob(p: float) -> str:
    # Conservative 4-bin mapping
    if p >= 0.65:
        return "very likely"
    if p >= 0.55:
        return "slightly likely"
    if p >= 0.35:
        return "not likely"
    return "not at all likely"


def main():
    if not DB.exists():
        print("Database not found: nba_bets.db")
        raise SystemExit(1)

    model_file = find_latest_model()
    if not model_file:
        print("No model artifact found in models/ — run training first.")
        raise SystemExit(1)

    model = joblib.load(model_file)
    conn = sqlite3.connect(str(DB))

    today = datetime.date.today().isoformat()
    df = load_games_for_date(conn, today)
    conn.close()

    if df.empty:
        print(f"No upcoming games with features found for date {today}.")
        raise SystemExit(0)

    # derived features
    df["home_pregame_rest_days"] = df["home_pregame_rest_days"].fillna(2.0)
    df["away_pregame_rest_days"] = df["away_pregame_rest_days"].fillna(2.0)
    df["streak_net_home"] = df["home_pregame_win_streak"] - df["home_pregame_loss_streak"]
    df["streak_net_away"] = df["away_pregame_win_streak"] - df["away_pregame_loss_streak"]
    df["streak_net_diff"] = df["streak_net_home"] - df["streak_net_away"]
    df["rest_days_diff"] = df["home_pregame_rest_days"] - df["away_pregame_rest_days"]

    X = df[[
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
    ]]

    probs = model.predict_proba(X)[:, 1]

    for idx, row in df.iterrows():
        p_home = float(probs[idx])
        p_away = 1.0 - p_home
        home = row["home_team_name"]
        away = row["visitor_team_name"]
        print(f"{home} vs {away} — game_id={int(row['game_id'])}")
        print(f"  {home} win probability: {p_home:.3f} ({label_from_prob(p_home)})")
        print(f"  {away} win probability: {p_away:.3f} ({label_from_prob(p_away)})")
        pick = home if p_home >= p_away else away
        conf = max(p_home, p_away)
        print(f"  Model pick: {pick} | confidence: {conf:.3f}\n")


if __name__ == '__main__':
    main()
