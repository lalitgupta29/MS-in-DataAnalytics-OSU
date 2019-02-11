import sqlite3

# connect to database
conn = sqlite3.connect('irs990.db')
c = conn.cursor()

print('\nTop 10 states by Average Revenue:')
# get the top 10 states by average revenue
for row in c.execute('''SELECT ROUND(AVG(total_rev), 2) as Avg_Revenue, 
                        bus_state as State 
                        FROM irs990 
                        WHERE State <> 'NA'
                        GROUP BY bus_state
                        ORDER BY Avg_Revenue DESC  
                        LIMIT 10'''):
    print(row)

conn.close()