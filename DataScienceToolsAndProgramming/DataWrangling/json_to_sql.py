import json
import sqlite3

conn = sqlite3.connect('irs990.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS indx
          (Object_Id text PRIMARY KEY NOT NULL,
          Tax_Period integer,
          DLN integer,
          Org_name text,
          Submit_On text,
          URL text)''')

c.execute('''CREATE TABLE IF NOT EXISTS irs990
          (Object_Id text PRIMARY KEY NOT NULL,
          tax_per_end_date text,
          tax_per_beg_date text,
          bus_ein text,
          bus_name text,
          bus_control text, 
          bus_phone text,
          bus_add text,
          bus_city text,
          bus_state text,
          bus_zip text,
          tax_year text,
          website text,
          total_rev int,
          total_exp int,
          net_asset_BOY int,
          net_asset_EOY)'''
         )
# conn.commit()

with open('index.json', 'r') as fp:
  index = json.load(fp)  

for idx in index:
  c.execute("INSERT INTO indx values (?,?,?,?,?,?)",
            (idx['ObjectId'], idx['Tax_Period'], idx['DLN'], 
            idx['org_Name'], idx['Submit_On'], idx['URL'])
           )

with open('irs990.json', 'r') as fp:
  filings = json.load(fp)
  
for filing in filings:
  c.execute("INSERT INTO irs990 values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (filing['ObjectId'], filing['tax_per_end_date'], filing['tax_per_beg_date'], 
            filing['bus_ein'], filing['bus_name'], filing['bus_control'], filing['bus_phone'], 
            filing['bus_add'], filing['bus_city'], filing['bus_state'], filing['bus_zip'], 
            filing['tax_year'], filing['website'], filing['total_rev'], filing['total_exp'], 
            filing['net_asset_BOY'], filing['net_asset_EOY']))  

conn.commit()
conn.close()

