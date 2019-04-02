import requests
import json as j
import sys

# get the symbol of stock from the user
#symbol = input("Please input the stock symbol: ")
symbol = sys.argv[1]

# create the url to get the price data throuh API
url = "https://api.iextrading.com/1.0/stock/" + symbol + "/price"
response = requests.get(url)
if response.status_code < 400:
    result = j.loads(response.text)
    print("Current price for stock '" + symbol + "' is: " + 
          '${:,.2f}'.format(result))
else:
    print("System did not receive a proper response from the server. Please"  
           + " check the input and try again")
    
# best price at close over the month
url = "https://api.iextrading.com/1.0/stock/"+ symbol +"/chart/1m"
response = requests.get(url)
if response.status_code < 400:
    close_price = []
    result = j.loads(response.text)
        
    for item in result:
        close_price.append(item["close"])
    
    highest_price = sorted(close_price, reverse = True)[0]
    highest_price = '${:,.2f}'.format(highest_price)
    print("Highest price at close for stock '" + symbol 
          + "' over the last month was: " + highest_price)    
else:
    print("System did not receive a proper response from the server. Please"  
           + " check the input and try again")

# best price at close over the year
url = "https://api.iextrading.com/1.0/stock/"+ symbol +"/chart/1y"
response = requests.get(url)
if response.status_code < 400:
    close_price = []
    result = j.loads(response.text)
        
    for item in result:
        close_price.append(item["close"])
    
    highest_price = sorted(close_price, reverse = True)[0]
    highest_price = '${:,.2f}'.format(highest_price)
    print("Highest price at close for stock '" + symbol 
          + "' over the last year was: " + highest_price)   
else:
    print("System did not receive a proper response from the server. Please"  
           + " check the input and try again")

# company names of the gainers
url = "https://api.iextrading.com/1.0/stock/market/list/gainers"
response = requests.get(url)
if response.status_code < 400:
    result = j.loads(response.text)
    print("\nBelow is the list of gainers:")
    for item in result:
        print(item["companyName"])
else:
    print("\nSystem could not get the list of gainers from the server.")
    
# Losers
url = "https://api.iextrading.com/1.0/stock/market/list/losers"
response = requests.get(url)
if response.status_code < 400:
    result = j.loads(response.text)
    print("\nBelow is the list of losers:")
    for item in result:
        print(item["companyName"])
else:
    print("\nSystem could not get the list of losers from the server.")

# crypto 
url = "https://api.iextrading.com/1.0/stock/market/crypto"
response = requests.get(url)
if response.status_code < 400:
    crypto = []
    result = j.loads(response.text)
    
    for item in result:
        crypto.append([item["companyName"],item["changePercent"]])
    
    # crypto with lowest percentage change
    print("\nCrypto currency with lowest percentage change: " 
          + sorted(crypto, key=lambda x: x[1])[0][0])
    
    # crypto with highest percentage change
    print("\nCrypto currency with highest percentage change: "
          + sorted(crypto, key=lambda x: -x[1])[0][0])
else: 
    print("System could not get the crypto currency info from the server.")
