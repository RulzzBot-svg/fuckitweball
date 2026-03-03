import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
import datetime

DB = Path("nba_bets.db")


def load_latest_run(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        "SELECT run_id, created_at, test_accuracy, test_log_loss, test_roc_auc, start_bankroll, final_bankroll, roi_pct FROM model_backtest_runs ORDER BY created_at DESC LIMIT 1"
    )
    return cur.fetchone()


def load_predictions(conn: sqlite3.Connection, run_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        f"SELECT * FROM model_game_predictions WHERE run_id = ? ORDER BY game_date", conn, params=(run_id,)
    )


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1 + (odds / 100)
    return 1 + (100 / abs(odds))


def find_upcoming_games(conn: sqlite3.Connection, date_iso: str) -> pd.DataFrame:
    q = '''
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
    LEFT JOIN teams ht ON g.home_team_id = ht.id
    LEFT JOIN teams vt ON g.visitor_team_id = vt.id
    LEFT JOIN team_game_features h ON g.id = h.game_id AND h.team_id = g.home_team_id
    LEFT JOIN team_game_features a ON g.id = a.game_id AND a.team_id = g.visitor_team_id
    WHERE g.home_score IS NULL
      AND g.date LIKE ?
    ORDER BY g.date
    '''
    return pd.read_sql_query(q, conn, params=(f"{date_iso}%",))


def load_model_file():
    models_dir = Path("models")
    if not models_dir.exists():
        return None
    models = sorted(models_dir.glob("*.pkl"), reverse=True)
    return models[0] if models else None


def predict_probs_for_df(model, df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)

    # compute derived features
    df = df.copy()
    df["home_pregame_rest_days"] = df["home_pregame_rest_days"].fillna(2.0)
    df["away_pregame_rest_days"] = df["away_pregame_rest_days"].fillna(2.0)
    df["streak_net_home"] = df["home_pregame_win_streak"] - df["home_pregame_loss_streak"]
    df["streak_net_away"] = df["away_pregame_win_streak"] - df["away_pregame_loss_streak"]
    df["streak_net_diff"] = df["streak_net_home"] - df["streak_net_away"]
    df["rest_days_diff"] = df["home_pregame_rest_days"] - df["away_pregame_rest_days"]

    features = [
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

    X = df[features]
    probs = model.predict_proba(X)[:, 1]
    return pd.Series(probs, index=df.index)


def main():
    st.title("NBA Bets — Phase 4 UI")

    if not DB.exists():
        st.error("Database not found (nba_bets.db). Run ingestion and feature build first.")
        return

    conn = sqlite3.connect(str(DB))

    latest = load_latest_run(conn)
    if latest:
        run_id, created_at, acc, ll, roc_auc, start_bankroll, final_bankroll, roi = latest
        st.subheader("Latest Backtest Run")
        st.write("Run ID:", run_id)
        st.write("Created:", created_at)
        st.metric("Test Accuracy", f"{acc:.4f}")
        st.metric("Log Loss", f"{ll:.4f}")
        st.metric("ROC AUC", f"{roc_auc:.4f}")
        st.metric("Start Bankroll", f"${start_bankroll:.2f}")
        st.metric("Final Bankroll", f"${final_bankroll:.2f}")
        st.metric("ROI %", f"{roi:.2f}%")
    else:
        st.info("No model runs found. Run `python train_backtest.py` to train and backtest.")

    # Predictions table and calibration from latest run
    if latest:
        df_preds = load_predictions(conn, run_id)
        st.subheader("Predictions Table")
        st.dataframe(df_preds)

        if not df_preds.empty:
            probs = df_preds["prob_home_win"].values
            y_true = df_preds["home_win"].values

            st.subheader("Calibration")
            frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10)
            fig, ax = plt.subplots()
            ax.plot(mean_pred, frac_pos, marker="o", label="Calibration")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.legend()
            st.pyplot(fig)

            st.write("Brier score:", f"{brier_score_loss(y_true, probs):.4f}")
            st.write("ROC AUC:", f"{roc_auc_score(y_true, probs):.4f}")

    st.subheader("Predict Upcoming Games")

    model_file = load_model_file()
    if model_file:
        st.success(f"Found model artifact: {model_file}")
        model = joblib.load(model_file)
    else:
        st.info("No model artifact found in models/. Run training to create one.")
        model = None

    col1, col2 = st.columns([1, 1])
    with col1:
        date_sel = st.date_input("Select date", value=datetime.date.today())
    with col2:
        if st.button("Load games for date"):
            # store in session state
            games_df = find_upcoming_games(conn, date_sel.isoformat())
            st.session_state["upcoming_games"] = games_df

    games_df = st.session_state.get("upcoming_games") if "upcoming_games" in st.session_state else None

    if games_df is None:
        st.info("Click 'Load games for date' to pull upcoming games (requires pregame features).")
    else:
        if games_df.empty:
            st.warning("No upcoming games with pregame features found for that date.")
        else:
            probs = predict_probs_for_df(model, games_df) if model is not None else pd.Series([None]*len(games_df))
            games_df = games_df.copy()
            games_df["prob_home_win"] = probs.values

            st.write("Games:")
            # allow user to input odds per game
            odds_inputs = {}
            for i, row in games_df.iterrows():
                gid = int(row["game_id"])
                home = row.get("home_team_name") or row.get("home_team_id")
                away = row.get("visitor_team_name") or row.get("visitor_team_id")
                st.markdown(f"**{home} vs {away}** — game_id: {gid}")
                if pd.notna(row.get("prob_home_win")):
                    st.write(f"Model: prob_home={row['prob_home_win']:.4f} prob_away={1-row['prob_home_win']:.4f}")
                col_a, col_b = st.columns(2)
                with col_a:
                    key_h = f"odds_home_{gid}"
                    odds_inputs[key_h] = st.text_input("Home odds (American)", key=key_h)
                with col_b:
                    key_a = f"odds_away_{gid}"
                    odds_inputs[key_a] = st.text_input("Away odds (American)", key=key_a)
                st.divider()

            if st.button("Compute edges for entered odds"):
                results = []
                for i, row in games_df.iterrows():
                    gid = int(row["game_id"])                    
                    p_home = row.get("prob_home_win")
                    if p_home is None or pd.isna(p_home):
                        continue
                    h_key = f"odds_home_{gid}"
                    a_key = f"odds_away_{gid}"
                    h_raw = st.session_state.get(h_key)
                    a_raw = st.session_state.get(a_key)
                    try:
                        if h_raw:
                            h_odds = int(h_raw)
                            dec = american_to_decimal(h_odds)
                            implied = 1.0 / dec
                            edge = p_home - implied
                            ev = p_home * dec - 1.0
                            results.append((gid, row.get("home_team_name"), row.get("visitor_team_name"), "home", h_odds, p_home, implied, edge, ev))
                        if a_raw:
                            a_odds = int(a_raw)
                            dec2 = american_to_decimal(a_odds)
                            implied2 = 1.0 / dec2
                            edge2 = (1.0 - p_home) - implied2
                            ev2 = (1.0 - p_home) * dec2 - 1.0
                            results.append((gid, row.get("home_team_name"), row.get("visitor_team_name"), "away", a_odds, 1.0 - p_home, implied2, edge2, ev2))
                    except Exception as e:
                        st.error(f"Invalid odds for game {gid}: {e}")

                if results:
                    res_df = pd.DataFrame(results, columns=["game_id", "home", "away", "side", "odds", "model_prob", "implied_prob", "edge", "ev_per_$1"])
                    st.subheader("Edges / EV")
                    st.dataframe(res_df)


if __name__ == "__main__":
    main()
