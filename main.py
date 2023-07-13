import datetime
import os
now = datetime.datetime.now()
print ("Current date and time : ")
print (now.strftime("%Y-%m-%d %H:%M:%S"))
from urllib.request import urlopen
import re as r
 
def getIP():
    d = str(urlopen('http://checkip.dyndns.com/')
            .read())
 
    return r.compile(r'Address: (\d+\.\d+\.\d+\.\d+)').search(d).group(1)
 
print('Your Public IPv4 is',getIP())