import sqlite3

# connect to database
conn = sqlite3.connect('irs990.db')
c = conn.cursor()

print('\nFirst 10 rows of Index table:')
# get first 20 rows of index table
for row in c.execute('''SELECT *
                        FROM indx
                        LIMIT 10'''):
    print(row)

print('\nFirst 10 rows of irs990 table:')
# get first 20 rows of irs990 table
for row in c.execute('''SELECT * 
                        from irs990
                        LIMIT 10'''):
    print(row)

conn.close()