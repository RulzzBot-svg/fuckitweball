import sqlite3
import pandas as pd
import sys
from pathlib import Path

DB = Path('nba_bets.db')
if not DB.exists():
    print('Database not found:', DB)
    sys.exit(1)

teams_to_match = [
    '76ers', 'sixers', 'grizzlies', 'grizz', 'timberwolves', 'thunder', 'bulls',
    'pelicans', 'lakers', 'spurs', 'raptors', 'gaspurs'
]

# Allow passing custom team substrings via args
if len(sys.argv) > 1:
    teams_to_match = [arg.lower() for arg in sys.argv[1:]]

conn = sqlite3.connect(str(DB))
query = '''
SELECT g.id AS game_id, g.date AS game_date, g.season,
       ht.full_name AS home_team, vt.full_name AS visitor_team,
       g.home_score, g.visitor_score
FROM games g
LEFT JOIN teams ht ON g.home_team_id = ht.id
LEFT JOIN teams vt ON g.visitor_team_id = vt.id
WHERE g.home_score IS NULL
ORDER BY g.date
'''
df = pd.read_sql_query(query, conn)
conn.close()

if df.empty:
    print('No upcoming games with null scores found in DB.')
    sys.exit(0)

# lower-case for matching
matches = []
for _, r in df.iterrows():
    home = (r['home_team'] or '').lower()
    away = (r['visitor_team'] or '').lower()
    for t in teams_to_match:
        if t in home or t in away:
            matches.append(r)
            break

if not matches:
    print('No matching upcoming games found for:', teams_to_match)
    print('\nAll upcoming games:')
    print(df.to_string(index=False))
else:
    print('Matching upcoming games:')
    out = pd.DataFrame(matches)
    print(out.to_string(index=False))
