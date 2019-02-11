import csv
import json

csv_rows = []
with open('irs990.csv', 'r') as fp:
  reader = csv.DictReader(fp)
  title = reader.fieldnames
  for row in reader:
    csv_rows.extend([{title[i]:row[title[i]] for i in range(len(title))}])

with open('irs990.json', "w") as fp:
    fp.write(json.dumps(csv_rows))

csv_rows = []
with open('index.csv', 'r') as fp:
  reader = csv.DictReader(fp)
  title = reader.fieldnames
  for row in reader:
    csv_rows.extend([{title[i]:row[title[i]] for i in range(len(title))}])

with open('index.json', "w") as fp:
    fp.write(json.dumps(csv_rows))
