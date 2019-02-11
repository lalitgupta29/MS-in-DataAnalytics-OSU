import csv
import json as j

with open('irs990.json', 'r') as fp:
  irs990 = j.load(fp)

with open('irs990.csv', 'w', newline='') as fp:
    fieldnames = irs990[0].keys()
    writer = csv.DictWriter(fp, fieldnames)
    writer.writeheader()
    for form in irs990:
        writer.writerow(form)

with open('index.json', 'r') as fp:
  index = j.load(fp)

with open('index.csv', 'w', newline='') as fp:
    fieldnames = index[0].keys()
    writer = csv.DictWriter(fp, fieldnames)
    writer.writeheader()
    for idx in index:
        writer.writerow(idx)

