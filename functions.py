import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import json
import csv
from bitarray import bitarray
from scipy import optimize

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


def cosf(x, A, nu):
    return A * np.cos(nu * x) # sine function with amplitude A and angular frequency

#parameters: original Doppler samples, original sampling period, starting point in seconds of the interpolated samples
# to be returned, number of interpolated samples needed, interpolated sampling period
#returns: interpolated doppler shifts
def GetDopplerShift(originalDopplerSamples, originalSamplingPeriod, startingTime, numInterpSamples, interpSamplingPeriod):

    SampleTime = (np.asarray(range(0,len(originalDopplerSamples))))*originalSamplingPeriod #create the x vector (original)
    InterpSampleTime = (np.asarray(range(0,numInterpSamples)))*interpSamplingPeriod #create the x vector (interpolated)
    InterpSampleTime = np.asarray([i+startingTime for i in InterpSampleTime])

    popt, pcov = optimize.curve_fit(cosf, SampleTime, originalDopplerSamples, p0=[3, 0.00001], full_output=False)

    return cosf(InterpSampleTime, popt[0], popt[1])


#Function to append a string containing bits into a bitarray type ba
def appendOnBA(ba,object):
    if(isinstance(object,str)):
        for c in object: ba.append(int(c))
    else:
        for c in object: ba.append(c)


#function to code a literals string into a binary string using ASCII 7bit
#useless for the user transmitter, but might be useful somehow
def stringToASCII7(message):
    bin_message=""

    for char in message:
        word=format(ord(char),"b")
        word=word.zfill(7)
        bin_message=bin_message+word

    return bin_message


#This function will create a message of ACK
#Receives the syncPattern, SVID as str and msgID as int
#Returns a bitarray
def createMessageACK(syncPattern, SVID, msgID):
    ACKpattern=(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    msg=bitarray()
    appendOnBA(msg, syncPattern)
    SVID=format(int(SVID),"b")
    while(len(SVID)<6):
        SVID='0'+SVID
    appendOnBA(msg, SVID)
    msgID=format(msgID,"b")
    while(len(msgID)<4):
        msgID='0'+msgID
    appendOnBA(msg, msgID)
    appendOnBA(msg,ACKpattern)
    informationBA=bitarray(msgID)
    appendOnBA(informationBA,ACKpattern)
    CRC=ComputeCRC(informationBA)
    appendOnBA(msg,CRC)
    appendOnBA(msg,'000000')

    return msg

#This function will create a message of NACK
#Receives the syncPattern, SVID as str and msgID as int
#Returns a bitarray
def createMessageNACK(syncPattern, SVID, msgID):
    ACKpattern=(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
    msg=bitarray()
    appendOnBA(msg, syncPattern)
    SVID=format(int(SVID),"b")
    while(len(SVID)<6):
        SVID='0'+SVID
    appendOnBA(msg, SVID)
    msgID=format(msgID,"b")
    while(len(msgID)<4):
        msgID='0'+msgID
    appendOnBA(msg, msgID)
    appendOnBA(msg,ACKpattern)
    informationBA=bitarray(msgID)
    appendOnBA(informationBA,ACKpattern)
    CRC=ComputeCRC(informationBA)
    appendOnBA(msg,CRC)
    appendOnBA(msg,'000000')

    return msg