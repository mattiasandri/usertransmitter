import json
import math
from bitarray import bitarray
from bitarray.util import int2ba
import createMessage



f = open('settings.ini')
settings = json.load(f)
 
for i in settings:
    print(i, end=': ')
    print(settings[i])
 
f.close() 

print(createMessage.createMessage('messagetobecodedinbinary',settings['SyncPattern'],settings['SVID']))