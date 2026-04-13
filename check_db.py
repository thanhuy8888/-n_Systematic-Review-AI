import sqlite3
import pprint

conn = sqlite3.connect('sr_data.db')
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
print("Tables:", tables)

for table in tables:
    count = conn.execute(f"SELECT COUNT(*) FROM {table[0]};").fetchone()[0]
    print(f"Table '{table[0]}': {count} entries")

conn.close()
