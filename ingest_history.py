import argparse
import os
import sqlite3
import time
from typing import Iterable

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.balldontlie.io/v1"
DEFAULT_DB_NAME = os.getenv("DB_NAME", "nba_bets.db")
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "12"))
API_KEY = os.getenv("BALLDONTLIE_API_KEY", "").strip()


def get_headers() -> dict:
    if not API_KEY:
        raise ValueError("BALLDONTLIE_API_KEY is not set. Add it to your .env file.")
    return {"Authorization": API_KEY}


def init_db(db_name: str = DEFAULT_DB_NAME) -> sqlite3.Connection:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY,
            full_name TEXT NOT NULL,
            abbreviation TEXT NOT NULL,
            conference TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY,
            date TEXT NOT NULL,
            season INTEGER NOT NULL,
            status TEXT,
            period INTEGER,
            postseason INTEGER NOT NULL DEFAULT 0,
            home_team_id INTEGER NOT NULL,
            visitor_team_id INTEGER NOT NULL,
            home_score INTEGER,
            visitor_score INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(home_team_id) REFERENCES teams(id),
            FOREIGN KEY(visitor_team_id) REFERENCES teams(id)
        )
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_games_season_date
        ON games (season, date)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_games_home_team
        ON games (home_team_id)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_games_visitor_team
        ON games (visitor_team_id)
        """
    )

    conn.commit()
    return conn


def _upsert_team(cursor: sqlite3.Cursor, team: dict) -> None:
    cursor.execute(
        """
        INSERT INTO teams (id, full_name, abbreviation, conference)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            full_name = excluded.full_name,
            abbreviation = excluded.abbreviation,
            conference = excluded.conference
        """,
        (
            team.get("id"),
            team.get("full_name"),
            team.get("abbreviation"),
            team.get("conference"),
        ),
    )


def _is_finished_game(game: dict) -> bool:
    status = str(game.get("status", "")).lower()
    return "final" in status


def fetch_and_store_games(conn: sqlite3.Connection, season: int) -> dict:
    cursor = conn.cursor()
    headers = get_headers()

    next_cursor = None
    total_seen = 0
    total_saved = 0
    total_finished = 0

    print(f"Starting ingestion for season {season}...")

    while True:
        params = {"seasons[]": season, "per_page": 100}
        if next_cursor:
            params["cursor"] = next_cursor

        response = requests.get(
            f"{BASE_URL}/games", headers=headers, params=params, timeout=30
        )

        if response.status_code == 429:
            print("Rate limit hit (429). Sleeping 60s before retry...")
            time.sleep(60)
            continue

        response.raise_for_status()
        payload = response.json()
        games = payload.get("data", [])

        if not games:
            print("No more games returned for this page.")

        for game in games:
            total_seen += 1

            home_team = game.get("home_team", {})
            visitor_team = game.get("visitor_team", {})

            if home_team.get("id"):
                _upsert_team(cursor, home_team)
            if visitor_team.get("id"):
                _upsert_team(cursor, visitor_team)

            if not _is_finished_game(game):
                continue

            total_finished += 1

            cursor.execute(
                """
                INSERT OR REPLACE INTO games
                (id, date, season, status, period, postseason, home_team_id, visitor_team_id, home_score, visitor_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game.get("id"),
                    game.get("date"),
                    game.get("season"),
                    game.get("status"),
                    game.get("period"),
                    int(bool(game.get("postseason", False))),
                    home_team.get("id"),
                    visitor_team.get("id"),
                    game.get("home_team_score"),
                    game.get("visitor_team_score"),
                ),
            )
            total_saved += 1

        conn.commit()
        print(
            f"Season {season} progress -> seen: {total_seen}, finished: {total_finished}, upserted: {total_saved}"
        )

        next_cursor = payload.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break

        time.sleep(REQUEST_DELAY_SECONDS)

    print(
        f"Completed season {season}. Seen={total_seen}, finished={total_finished}, upserted={total_saved}"
    )
    return {
        "season": season,
        "seen": total_seen,
        "finished": total_finished,
        "upserted": total_saved,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize SQLite DB and ingest historical NBA games from BALLDONTLIE"
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
        required=True,
        help="One or more seasons to ingest, e.g. --seasons 2024 2025",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conn = init_db(args.db)
    try:
        summaries = []
        for season in args.seasons:
            summaries.append(fetch_and_store_games(conn, season))

        total_upserted = sum(item["upserted"] for item in summaries)
        print(f"Initial data load complete. Total upserted rows: {total_upserted}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
