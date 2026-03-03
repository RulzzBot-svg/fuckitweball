import sqlite3
from pathlib import Path
import pandas as pd
import joblib
import datetime
import re
import sys

DB = Path("nba_bets.db")
MODELS_DIR = Path("models")

def find_team(conn, name_substr):
    cur = conn.cursor()
    s = name_substr.lower()
    cur.execute("SELECT id, full_name, abbreviation FROM teams")
    rows = cur.fetchall()
    matches = [r for r in rows if s in (r[1] or '').lower() or s in (r[2] or '').lower()]
    return matches


def choose_unique(matches):
    if len(matches) == 1:
        return matches[0][0]
    return None


def get_season(conn):
    cur = conn.cursor()
    cur.execute("SELECT MAX(season) FROM games")
    r = cur.fetchone()
    return int(r[0]) if r and r[0] else datetime.date.today().year


def insert_game(conn, game_date_iso, season, home_id, away_id):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO games (date, season, status, period, postseason, home_team_id, visitor_team_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (game_date_iso, season, None, None, 0, home_id, away_id),
    )
    conn.commit()
    return cur.lastrowid


def last_feature_row(conn, team_id, before_date_iso):
    cur = conn.cursor()
    cur.execute(
        "SELECT game_date, postgame_win_streak, postgame_loss_streak FROM team_game_features WHERE team_id = ? AND game_date < ? ORDER BY game_date DESC LIMIT 1",
        (team_id, before_date_iso),
    )
    return cur.fetchone()


def calc_rest_days(last_date_str, current_date_str):
    if not last_date_str:
        return None
    last = datetime.date.fromisoformat(last_date_str)
    cur = datetime.date.fromisoformat(current_date_str.split('T')[0])
    delta = (cur - last).days - 1
    return max(delta, 0)


def insert_pregame_feature(conn, game_id, season, game_date_iso, team_id, opp_id, is_home):
    last = last_feature_row(conn, team_id, game_date_iso)
    if last:
        last_game_date, post_win, post_loss = last
        pre_win = post_win
        pre_loss = post_loss
        pre_rest = calc_rest_days(last_game_date, game_date_iso)
    else:
        pre_win = 0
        pre_loss = 0
        pre_rest = None

    cur = conn.cursor()
    # If a feature row already exists for this game/team, skip
    cur.execute("SELECT 1 FROM team_game_features WHERE game_id = ? AND team_id = ?", (game_id, team_id))
    if cur.fetchone():
        return

    cur.execute(
        "INSERT OR REPLACE INTO team_game_features (game_id, season, game_date, team_id, opponent_team_id, is_home, points_for, points_against, won, point_margin, pregame_win_streak, pregame_loss_streak, pregame_rest_days, postgame_win_streak, postgame_loss_streak) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            game_id,
            season,
            game_date_iso.split('T')[0],
            team_id,
            opp_id,
            1 if is_home else 0,
            0,
            0,
            0,
            0,
            pre_win,
            pre_loss,
            pre_rest,
            pre_win,
            pre_loss,
        ),
    )
    conn.commit()


def find_latest_model():
    if not MODELS_DIR.exists():
        return None
    models = sorted(MODELS_DIR.glob("*.pkl"), reverse=True)
    return models[0] if models else None


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1 + (odds / 100)
    return 1 + (100 / abs(odds))

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


def run_predictions(model_path, conn, game_ids):
    model = joblib.load(model_path)
    out_rows = []
    for gid in game_ids:
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
        WHERE g.id = ?
        """
        df = pd.read_sql_query(q, conn, params=(gid,))
        if df.empty:
            print(f"No feature rows found for game {gid}")
            continue
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
        prob_home = float(model.predict_proba(X)[:, 1][0])
        out_rows.append((gid, df.loc[0, 'game_date'], df.loc[0, 'home_team_id'], df.loc[0, 'visitor_team_id'], prob_home))
    return out_rows


def normalize_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = s.replace('gaspurs', 'spurs')
    s = s.replace('grizziles', 'grizzlies')
    s = s.replace('lakersmes', 'lakers')
    return s


def main():
    if len(sys.argv) < 2:
        print("Usage: python add_and_predict.py \"Home vs Away\" [\"Home vs Away\"]...")
        print("Example: python add_and_predict.py \"Spurs vs 76ers\" \"Grizzlies vs Timberwolves\"")
        raise SystemExit(1)

    matchups = sys.argv[1:]

    if not DB.exists():
        print("Database not found:", DB)
        raise SystemExit(1)

    conn = sqlite3.connect(str(DB))
    cur = conn.cursor()

    season = get_season(conn)
    today_iso = datetime.date.today().isoformat() + 'T00:00:00'

    inserted_game_ids = []

    for m in matchups:
        parts = re.split(r"vs|v|vs\.|@", m, flags=re.IGNORECASE)
        if len(parts) < 2:
            print("Could not parse matchup:", m)
            continue
        home_raw = normalize_name(parts[0].strip())
        away_raw = normalize_name(parts[1].strip())

        home_matches = find_team(conn, home_raw)
        away_matches = find_team(conn, away_raw)

        home_id = choose_unique(home_matches)
        away_id = choose_unique(away_matches)

        if not home_id:
            print(f"Ambiguous or missing home team for '{parts[0].strip()}':", home_matches)
        if not away_id:
            print(f"Ambiguous or missing away team for '{parts[1].strip()}':", away_matches)
        if not home_id or not away_id:
            print("Skipping matchup due to ambiguity. Try a different substring or full team name.")
            continue

        gid = insert_game(conn, today_iso, season, home_id, away_id)
        print(f"Inserted game {gid}: {home_matches[0][1]} vs {away_matches[0][1]} on {today_iso}")
        inserted_game_ids.append(gid)

        # insert pregame features for both teams
        insert_pregame_feature(conn, gid, season, today_iso, home_id, away_id, True)
        insert_pregame_feature(conn, gid, season, today_iso, away_id, home_id, False)

    if not inserted_game_ids:
        print("No games inserted. Exiting.")
        conn.close()
        raise SystemExit(0)

    model_file = find_latest_model()
    if not model_file:
        print("No model artifact found in models/. Run training first.")
        conn.close()
        raise SystemExit(1)

    results = run_predictions(model_file, conn, inserted_game_ids)
    conn.close()

    if not results:
        print("No predictions produced.")
        raise SystemExit(0)

    print('\nPredictions:')
    for gid, gdate, hid, vid, p_home in results:
        print(f"Game {gid} | date={gdate} | home_id={hid} away_id={vid} | prob_home={p_home:.4f} prob_away={1-p_home:.4f}")

if __name__ == '__main__':
    main()
