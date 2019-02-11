This folder contains the Data Wrnagling Project.

The project was done to pull the IRS 990 filings available on Amazon Web Services and accessed via making http calls to it.
The filings are available in XML format for each begininng in 2011. The list of XML links is available in json and csv format
accessed vis http call. 

The end objective is to identify the top 10 states in terms of average revenue. In order to do this, data is pulled and parsed 
from the web and put into json format. Then moved from json to sqlite database and then is queried from the database. 

PS: Because I was running this on my local computer, I was not able to pull all the data. So I restricted my calls to 30K.
