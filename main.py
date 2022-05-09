import json
import math
from bitarray import bitarray
from bitarray.util import int2ba
import ASCII_7


f = open('settings.ini')
settings = json.load(f)
 
for i in settings:
    print(i, end=': ')
    print(settings[i])
 
f.close() 


print(ASCII_7.stringToASCII7("trythis"))