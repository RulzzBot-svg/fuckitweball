import sqlite3
import json

DB = 'nba_bets.db'

def main():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    out = {}
    try:
        # Check tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        out['tables'] = tables

        if 'model_backtest_runs' in tables:
            cur.execute('SELECT COUNT(*) FROM model_backtest_runs')
            out['model_backtest_runs_count'] = cur.fetchone()[0]
            cur.execute('''
                SELECT run_id, created_at, test_accuracy, test_log_loss, test_roc_auc, start_bankroll, final_bankroll, roi_pct
                FROM model_backtest_runs
                ORDER BY created_at DESC
                LIMIT 1
            ''')
            last = cur.fetchone()
            out['last_run'] = last
        else:
            out['model_backtest_runs_count'] = 0
            out['last_run'] = None

        if 'model_game_predictions' in tables:
            cur.execute('SELECT COUNT(*) FROM model_game_predictions')
            out['model_game_predictions_count'] = cur.fetchone()[0]
        else:
            out['model_game_predictions_count'] = 0

    finally:
        conn.close()

    print(json.dumps(out, default=str))

if __name__ == '__main__':
    main()
