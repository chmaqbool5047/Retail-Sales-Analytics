import sqlite3
import pandas as pd

# CSV read کرو
df = pd.read_csv('retail_sales_dataset.csv')

# SQLite database بناؤ
conn = sqlite3.connect('sales_analytics.db')

# Table بناؤ
df.to_sql('sales', conn, if_exists='replace', index=False)

print('✅ Database Successfully Ready!')
conn.close()
