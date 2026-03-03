import argparse
import os
import sqlite3
from datetime import date, datetime

from dotenv import load_dotenv

load_dotenv()

DEFAULT_DB_NAME = os.getenv("DB_NAME", "nba_bets.db")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build team-level modeling features (streaks, margins, rest days)"
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_NAME,
        help="SQLite DB file name/path (default: nba_bets.db or DB_NAME from .env)",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        help="Optional season filter, e.g. --seasons 2024 2025",
    )
    return parser.parse_args()


def init_feature_table(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_game_features (
            game_id INTEGER NOT NULL,
            season INTEGER NOT NULL,
            game_date TEXT NOT NULL,
            team_id INTEGER NOT NULL,
            opponent_team_id INTEGER NOT NULL,
            is_home INTEGER NOT NULL,
            points_for INTEGER NOT NULL,
            points_against INTEGER NOT NULL,
            won INTEGER NOT NULL,
            point_margin INTEGER NOT NULL,
            pregame_win_streak INTEGER NOT NULL,
            pregame_loss_streak INTEGER NOT NULL,
            pregame_rest_days INTEGER,
            postgame_win_streak INTEGER NOT NULL,
            postgame_loss_streak INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (game_id, team_id),
            FOREIGN KEY (game_id) REFERENCES games(id),
            FOREIGN KEY (team_id) REFERENCES teams(id),
            FOREIGN KEY (opponent_team_id) REFERENCES teams(id)
        )
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_team_features_team_date
        ON team_game_features (team_id, game_date)
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_team_features_season
        ON team_game_features (season)
        """
    )

    conn.commit()


def parse_game_date(raw_date: str) -> date:
    date_part = raw_date.split("T")[0]
    return datetime.fromisoformat(date_part).date()


def get_seasons_to_build(conn: sqlite3.Connection, seasons_filter: list[int] | None) -> list[int]:
    cur = conn.cursor()
    if seasons_filter:
        return sorted(set(seasons_filter))

    cur.execute("SELECT DISTINCT season FROM games ORDER BY season")
    return [row[0] for row in cur.fetchall()]


def clear_existing_rows(conn: sqlite3.Connection, seasons: list[int]) -> None:
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in seasons)
    cur.execute(f"DELETE FROM team_game_features WHERE season IN ({placeholders})", seasons)
    conn.commit()


def load_games_for_season(conn: sqlite3.Connection, season: int) -> list[tuple]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, date, season, home_team_id, visitor_team_id, home_score, visitor_score
        FROM games
        WHERE season = ?
          AND home_score IS NOT NULL
          AND visitor_score IS NOT NULL
        ORDER BY date ASC, id ASC
        """,
        (season,),
    )
    return cur.fetchall()


def calc_rest_days(last_game_date: date | None, current_game_date: date) -> int | None:
    if last_game_date is None:
        return None

    delta_days = (current_game_date - last_game_date).days - 1
    return max(delta_days, 0)


def build_features_for_season(conn: sqlite3.Connection, season: int) -> int:
    games = load_games_for_season(conn, season)
    if not games:
        print(f"Season {season}: no finished games found.")
        return 0

    team_state: dict[int, dict] = {}
    cur = conn.cursor()

    inserted_rows = 0

    for game_id, raw_date, game_season, home_team_id, visitor_team_id, home_score, visitor_score in games:
        game_date = parse_game_date(raw_date)

        participants = [
            {
                "team_id": home_team_id,
                "opp_id": visitor_team_id,
                "is_home": 1,
                "points_for": home_score,
                "points_against": visitor_score,
            },
            {
                "team_id": visitor_team_id,
                "opp_id": home_team_id,
                "is_home": 0,
                "points_for": visitor_score,
                "points_against": home_score,
            },
        ]

        for row in participants:
            team_id = row["team_id"]
            won = int(row["points_for"] > row["points_against"])
            point_margin = row["points_for"] - row["points_against"]

            state = team_state.setdefault(
                team_id,
                {
                    "last_game_date": None,
                    "win_streak": 0,
                    "loss_streak": 0,
                },
            )

            pregame_win_streak = state["win_streak"]
            pregame_loss_streak = state["loss_streak"]
            pregame_rest_days = calc_rest_days(state["last_game_date"], game_date)

            if won:
                postgame_win_streak = pregame_win_streak + 1
                postgame_loss_streak = 0
            else:
                postgame_win_streak = 0
                postgame_loss_streak = pregame_loss_streak + 1

            cur.execute(
                """
                INSERT OR REPLACE INTO team_game_features (
                    game_id,
                    season,
                    game_date,
                    team_id,
                    opponent_team_id,
                    is_home,
                    points_for,
                    points_against,
                    won,
                    point_margin,
                    pregame_win_streak,
                    pregame_loss_streak,
                    pregame_rest_days,
                    postgame_win_streak,
                    postgame_loss_streak
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    game_season,
                    game_date.isoformat(),
                    team_id,
                    row["opp_id"],
                    row["is_home"],
                    row["points_for"],
                    row["points_against"],
                    won,
                    point_margin,
                    pregame_win_streak,
                    pregame_loss_streak,
                    pregame_rest_days,
                    postgame_win_streak,
                    postgame_loss_streak,
                ),
            )

            state["last_game_date"] = game_date
            state["win_streak"] = postgame_win_streak
            state["loss_streak"] = postgame_loss_streak
            inserted_rows += 1

    conn.commit()
    print(f"Season {season}: wrote {inserted_rows} team-game feature rows.")
    return inserted_rows


def main() -> None:
    args = parse_args()

    conn = sqlite3.connect(args.db)
    try:
        init_feature_table(conn)
        seasons = get_seasons_to_build(conn, args.seasons)

        if not seasons:
            print("No seasons found in games table. Run ingestion first.")
            return

        clear_existing_rows(conn, seasons)

        total_rows = 0
        for season in seasons:
            total_rows += build_features_for_season(conn, season)

        print(f"Feature build complete. Total rows written: {total_rows}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
