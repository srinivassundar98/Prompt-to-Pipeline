# build_db.py
import sqlite3, random
conn = sqlite3.connect("demo.db")
cur = conn.cursor()

cur.execute("CREATE TABLE IF NOT EXISTS customers(id INTEGER PRIMARY KEY, name TEXT, city TEXT)")
cur.execute("CREATE TABLE IF NOT EXISTS orders(id INTEGER PRIMARY KEY, customer_id INT, amount REAL, ts TEXT)")
names = ["Ada", "Byron", "Grace", "Linus"]
cities = ["NYC", "Austin", "SF", "Boston"]
for i in range(20):
    cur.execute("INSERT INTO customers(name, city) VALUES(?,?)",
                (random.choice(names), random.choice(cities)))
    cur.execute("INSERT INTO orders(customer_id, amount, ts) VALUES(?,?, date('now','-{} days'))"
                .format(i), (i + 1, round(random.random()*500,2)))
conn.commit()
conn.close()
