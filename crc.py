from bitarray import bitarray
import numpy
from binPolynomial import *
#documentation is at page 53

def ComputeCRC(informationBA):
	PX=(1,0,0,1,0,1,0,1,1,1,0,1,1,1,0,0,0,1,0,0,0,0,0,1)
	Xplus1=(1,1)
	
	GX=BinPolyMul(PX,Xplus1)
	
	informationList=[]
	for i in range(len(informationBA)):
		informationList.append(informationBA.pop())
	
	mX = tuple(informationList)
	
	X24=(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1)
	mX24=BinPolyMul(mX,X24)
	
	[Q,R]=BinPolyDiv(mX24,GX)

	R=list(R)

	while(len(R)<24):
		R.insert(0,0)

	R=tuple(R)
	
	return R


