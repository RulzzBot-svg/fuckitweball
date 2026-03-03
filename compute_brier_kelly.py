import sqlite3
import argparse
import pandas as pd
import math
from pathlib import Path

DB = Path("nba_bets.db")


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1 + (odds / 100)
    return 1 + (100 / abs(odds))


def get_latest_run(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("SELECT run_id, created_at FROM model_backtest_runs ORDER BY created_at DESC LIMIT 1")
    return cur.fetchone()


def load_predictions(conn: sqlite3.Connection, run_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM model_game_predictions WHERE run_id = ? ORDER BY game_date", conn, params=(run_id,)
    )


def brier_score(probs, truths):
    probs = pd.Series(probs)
    truths = pd.Series(truths)
    return float(((probs - truths) ** 2).mean())


def kelly_fraction(p: float, decimal_odds: float) -> float:
    # b = decimal_odds - 1
    b = decimal_odds - 1.0
    q = 1.0 - p
    if b <= 0:
        return 0.0
    num = b * p - q
    den = b
    f = num / den
    return max(f, 0.0)


def main():
    parser = argparse.ArgumentParser(description="Compute Brier and Kelly for latest run")
    parser.add_argument("--db", default=str(DB))
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--fractional_kelly", type=float, default=0.25, help="Fraction of Kelly to stake")
    parser.add_argument("--top", type=int, default=10, help="Top N bets to show by Kelly fraction")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        latest = get_latest_run(conn)
        if not latest:
            print("No model runs found in DB.")
            return
        run_id, created_at = latest
        print(f"Latest run: {run_id} | created: {created_at}")

        df = load_predictions(conn, run_id)
        if df.empty:
            print("No predictions for the latest run")
            return

        # Compute Brier score for home-win probability vs actual
        brier = brier_score(df['prob_home_win'], df['home_win'])

        # per-row Kelly for the pick side (use odds column if present)
        kellys = []
        rows = []
        for _, r in df.iterrows():
            pick = r['pick_side']
            if pd.isna(r.get('odds')):
                # skip if no market odds saved
                kf = None
                stake = None
            else:
                try:
                    odds = int(r['odds'])
                    dec = american_to_decimal(odds)
                    p = r['prob_home_win'] if pick == 'home' else r['prob_away_win']
                    kf = kelly_fraction(p, dec)
                    stake = kf * args.bankroll
                except Exception:
                    kf = None
                    stake = None

            rows.append({
                'game_id': r['game_id'],
                'game_date': r['game_date'],
                'pick': pick,
                'model_prob': r['prob_home_win'] if pick == 'home' else r['prob_away_win'],
                'odds': r.get('odds'),
                'kelly_frac': kf,
                'kelly_stake': stake,
            })
            if kf is not None and kf > 0:
                kellys.append(kf)

        stats = {
            'brier_score': brier,
            'total_predictions': len(df),
            'bets_with_odds': sum(1 for r in rows if r['odds'] is not None),
            'bets_positive_kelly': sum(1 for r in rows if r['kelly_frac'] is not None and r['kelly_frac'] > 0),
            'avg_positive_kelly_frac': float(sum(kellys) / len(kellys)) if kellys else 0.0,
        }

        print('\nSummary:')
        print(f"  Brier score: {stats['brier_score']:.4f}")
        print(f"  Total predictions: {stats['total_predictions']}")
        print(f"  Predictions with odds: {stats['bets_with_odds']}")
        print(f"  Bets with positive Kelly: {stats['bets_positive_kelly']}")
        print(f"  Avg positive Kelly frac: {stats['avg_positive_kelly_frac']:.4f}")

        # show top N by Kelly fraction
        df_rows = pd.DataFrame(rows)
        df_pos = df_rows[df_rows['kelly_frac'].notna() & (df_rows['kelly_frac'] > 0)].copy()
        if df_pos.empty:
            print('\nNo positive Kelly bets found in this run (or no odds were recorded).')
            return

        df_pos.sort_values('kelly_frac', ascending=False, inplace=True)
        topn = df_pos.head(args.top)

        print('\nTop recommended bets by Kelly fraction:')
        for _, r in topn.iterrows():
            fk = r['kelly_frac']
            stake = r['kelly_stake']
            print(f"  Game {int(r['game_id'])} | pick={r['pick']} | prob={r['model_prob']:.3f} | odds={r['odds']} | kelly={fk:.4f} | stake(${stake:.2f})")

        # show fractional Kelly stakes
        if args.fractional_kelly and args.fractional_kelly > 0:
            print(f"\nApplying fractional Kelly: {args.fractional_kelly}x")
            for _, r in topn.iterrows():
                fk = r['kelly_frac']
                stake = (fk * args.fractional_kelly) * args.bankroll if fk is not None else None
                print(f"  Game {int(r['game_id'])} | fractional stake(${stake:.2f})")

    finally:
        conn.close()


if __name__ == '__main__':
    main()
