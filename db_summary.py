import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "nba_bets.db")


def main() -> None:
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM teams")
    teams = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM games")
    games = cur.fetchone()[0]

    cur.execute(
        """
        SELECT season, COUNT(*)
        FROM games
        GROUP BY season
        ORDER BY season
        """
    )
    by_season = cur.fetchall()

    print(f"DB: {DB_NAME}")
    print(f"Teams: {teams}")
    print(f"Games: {games}")
    print("Games by season:")
    for season, count in by_season:
        print(f"  {season}: {count}")

    conn.close()


if __name__ == "__main__":
    main()
