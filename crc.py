from numpy.polynomial import Polynomial
from bitarray import bitarray
from numpy.polynomial.polynomial import *
import numpy

#documentation is at page 53

informationBA=bitarray(34)

infLen=len(informationBA)
PX = Polynomial([1,0,0,1,0,1,0,1,1,1,0,1,1,1,0,0,0,1,0,0,0,0,0,1])
Xplus1 = Polynomial([1,1])
GX = Xplus1*PX

informationList=[]
for i in range(infLen):
	informationList.append(informationBA.pop())

mX = Polynomial(informationList)

X24 = Polynomial([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
mX24 = mX*X24
[Q,R]=polydiv(mX24.coef,GX.coef)

print(R)