# import required libraries
import requests
import csv
import json as j
import xml.etree.ElementTree as ET

# default variables
links = []
index = []
irs990_forms = []
ns = {'ns': 'http://www.irs.gov/efile'}
year_list = [str(f) for f in range(2011, 2019)]
counter = 0
no_of_records = 30000

# get links for individual filings and save in links
for year in year_list:
  url = 'https://s3.amazonaws.com/irs-form-990/index_' + year + '.json'

  if counter >= no_of_records:
    break

  try:
    print("getting data for year: " + year)
    resp = requests.get(url)
    print('Got data for year: ' + year)
  except:
    print("Could not get data for year: " + year)
    continue

  print('Loading data for year: ' + year)

  forms_index = j.loads(resp.text)

  for filing in forms_index['Filings' + year]:
    # get IRS 990 form data for each filing listed in index
    try:
      resp = requests.get(filing['URL'])
    except:
      continue
    
    root = ET.fromstring(resp.content)

    # if the form 990 or 990EZ are in the filing pull the data else
    # skip the filing
    if root.find('ns:ReturnData',ns).find('ns:IRS990',ns) != None:
      irs990 =  root.find('ns:ReturnData',ns).find('ns:IRS990',ns)
      if irs990.find('ns:TotalRevenueCurrentYear',ns) != None:
        tot_rev = irs990.find('ns:TotalRevenueCurrentYear',ns).text
      else:
        tot_rev = 'NA'
      if irs990.find('ns:TotalExpensesCurrentYear',ns) != None:
        tot_exp = irs990.find('ns:TotalExpensesCurrentYear',ns).text
      else:
        tot_exp = 'NA'
      if irs990.find('ns:NetAssetsOrFundBalancesBOY',ns) != None:
        net_asset_BOY = irs990.find('ns:NetAssetsOrFundBalancesBOY',ns).text
      else:
        net_asset_BOY = 'NA'
      if irs990.find('ns:NetAssetsOrFundBalancesEOY',ns) != None:
        net_asset_EOY = irs990.find('ns:NetAssetsOrFundBalancesEOY',ns).text
      else:
        net_asset_EOY = 'NA'
      if irs990.find('ns:WebSite',ns) != None:
        website = irs990.find('ns:WebSite', ns).text
      else:
        website = 'NA'
    elif root.find('ns:ReturnData',ns).find('ns:IRS990EZ',ns) != None:
      irs990 = root.find('ns:ReturnData',ns).find('ns:IRS990EZ',ns)
      if irs990.find('ns:TotalRevenue',ns) != None:
        tot_rev = irs990.find('ns:TotalRevenue',ns).text
      else:
        tot_rev = 'NA'
      if irs990.find('ns:TotalExpenses',ns) != None:
        tot_exp = irs990.find('ns:TotalExpenses',ns).text
      else:
        tot_exp = 'NA'
      if irs990.find('ns:NetAssetsOrFundBalancesBOY',ns) != None:
        net_asset_BOY = irs990.find('ns:NetAssetsOrFundBalancesBOY',ns).text
      else:
        net_asset_BOY = 'NA'
      if irs990.find('ns:NetAssetsOrFundBalancesEOY',ns) != None:
        net_asset_EOY = irs990.find('ns:NetAssetsOrFundBalancesEOY',ns).text
      else:
        net_asset_EOY = 'NA'
      if irs990.find('ns:WebSite',ns) != None:
        website = irs990.find('ns:WebSite', ns).text
      else:
        website = 'NA'
    else:
      continue

    if root.find('ns:ReturnHeader', ns) != None:
      ret_head = root.find('ns:ReturnHeader', ns)
      if ret_head.find('ns:TaxPeriodEndDate', ns) != None:
        tax_per_end_date = ret_head.find('ns:TaxPeriodEndDate', ns).text
      else:
        tax_per_end_date = 'NA'

      if ret_head.find('ns:TaxPeriodBeginDate', ns) != None:
        tax_per_beg_date = ret_head.find('ns:TaxPeriodBeginDate', ns).text
      else:
        tax_per_beg_date = 'NA'

      if ret_head.find('ns:Filer',ns) != None:
        filer = ret_head.find('ns:Filer',ns)
        if filer.find('ns:EIN', ns) != None:
          bus_ein = filer.find('ns:EIN', ns).text
        else:
          bus_ein = 'NA'

        if filer.find('ns:Name',ns).find('ns:BusinessNameLine1',ns) != None:
          bus_name = filer.find('ns:Name',ns).find('ns:BusinessNameLine1',ns).text
        else:
          bus_name = 'NA'

        if filer.find('ns:NameControl', ns) != None:
          bus_control = filer.find('ns:NameControl', ns).text
        else:
          bus_control = 'NA'

        if filer.find('ns:Phone', ns) != None:
          bus_phone = filer.find('ns:Phone', ns).text
        else:
          bus_phone = 'NA'

        if filer.find('ns:USAddress', ns) != None:
          if filer.find('ns:USAddress', ns).find('ns:AddressLine1',ns) != None:
            bus_add = filer.find('ns:USAddress', ns).find('ns:AddressLine1',ns).text
            bus_add = bus_add.replace(',', '')  # remove commas from address
          else:
            bus_add = 'NA'
        else:
          bus_add = 'NA'

        if filer.find('ns:USAddress', ns) != None:
          if filer.find('ns:USAddress', ns).find('ns:City',ns) != None:
            bus_city = filer.find('ns:USAddress', ns).find('ns:City',ns).text
          else:
            bus_city = 'NA'
        else:
          bus_city = 'NA'

        if filer.find('ns:USAddress', ns) != None:
          if filer.find('ns:USAddress', ns).find('ns:State',ns) != None:
            bus_state = filer.find('ns:USAddress', ns).find('ns:State',ns).text
          else:
            bus_state = 'NA'
        else:
          bus_state = 'NA'

        if filer.find('ns:USAddress', ns) != None:
          if filer.find('ns:USAddress', ns).find('ns:Zip',ns) != None:
            bus_zip = filer.find('ns:USAddress', ns).find('ns:Zip',ns).text
          elif filer.find('ns:USAddress', ns).find('ns:ZIPCode',ns) != None:
           bus_zip = filer.find('ns:USAddress', ns).find('ns:ZIPCode',ns).text
          else:
            bus_zip = 'NA'
        else:
          bus_zip = 'NA'

      if ret_head.find('ns:TaxYear', ns) != None:
        tax_year = ret_head.find('ns:TaxYear', ns).text
      else:
        tax_year = 'NA'
    else:
      continue

    # append filing data from IRS990 to list irs990
    irs990_forms.append({'ObjectId': filing['ObjectId'],
                         'tax_per_end_date': tax_per_end_date,
                         'tax_per_beg_date': tax_per_beg_date,
                         'bus_ein': bus_ein,
                         'bus_name': bus_name,
                         'bus_control': bus_control,
                         'bus_phone': bus_phone,
                         'bus_add': bus_add,
                         'bus_city': bus_city,
                         'bus_state': bus_state,
                         'bus_zip': bus_zip,
                         'tax_year': tax_year,
                         'website': website,
                         'total_rev': tot_rev,
                         'total_exp': tot_exp,
                         'net_asset_BOY': net_asset_BOY,
                         'net_asset_EOY': net_asset_EOY
                        })

    # append individual filing data from index file to list index
    index.append({'ObjectId': filing['ObjectId'],
                  'Tax_Period': filing['TaxPeriod'],
                  'DLN': filing['DLN'],
                  'org_Name': filing['OrganizationName'],
                  'Submit_On': filing['SubmittedOn'],
                  'URL': filing['URL']
                 })

    if len(index) % 20 == 0:
      print('Total records processed: '+str(len(index)))

    # increment the counter
    counter += 1
    if counter >= no_of_records:
      break
      
  print('Index length: '+ str(len(index)))
  print('IRS990 length: ' + str(len(irs990_forms)))

print('Writing to csv file "irs990"')
# save to csv file
with open('irs990.csv', 'w', newline='') as fp:
  fieldnames = irs990_forms[0].keys()
  writer = csv.DictWriter(fp, fieldnames)
  writer.writeheader()
  for row in irs990_forms:
    writer.writerow(row)
print('Write complete.')

print('Writing to csv file "index"')
with open('index.csv', 'w', newline='') as fp:
  fieldnames = index[0].keys()
  writer = csv.DictWriter(fp, fieldnames)
  writer.writeheader()
  for row in index:
    writer.writerow(row)
print('Write complete.')

print('Writing to json file "irs990"')
# save to json file
with open('irs990.json', 'w') as fp:
  j.dump(irs990_forms, fp, indent=2)
print('Write complete.')

print('Writing to json file "index"')
with open('index.json', 'w') as fp:
  j.dump(index, fp, indent=2)
print('Write complete.')
print('End of script')



