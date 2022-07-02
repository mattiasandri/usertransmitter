import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from math import pi
from scipy.signal import upfirdn
import json
import csv
from bitarray import bitarray
from scipy import optimize

#receives a bitarray containing the information bits, in our case only
#the ACK or NACK strings
#returns a tuple containng the CRC calculated
#this function will be user by the createdMessage*** functions
#details are at page 53
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

# simple conversion from hexadecimal to binary
def hex2bin(h):
    integer = int(h, 16)
    binary = bin(integer)[2::]     #discarding the first two elements (i.e. 0b, not needed)
    binary = np.array([int(x) for x in binary])    #the binary representation is a numpy array of integers
    return binary

def conversion(prn):
    new_prn = np.zeros(len(prn))
    for i in range(len(prn)):
        if prn[i] == 0:
            new_prn[i] = 1
        elif prn[i] == 1:
            new_prn[i] = -1
    return np.array(new_prn, int)   # the prns are stored in a numpy array of integers

#this function plots the original PRNs
def plot_prn_zeros(N, prn):
    
    Rc = 1.023e06   # chip rate
    Tc = 1 / Rc     # chip period 
    
    t = np.arange(0, N*Tc, Tc)
    
    xticks = t
    yticks = [0, 1]
    centers = 0.5*(xticks[1:] + xticks[:-1]) 
    
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1,1,1)
    ax.step(t, prn[0:N], where='post', color='red', lw=3)
    ax.set_xticks(t)
    ax.set_yticks(yticks)
    title = "Prn to plot: " + str(prn[0:N])
    ax.set_title(title, fontsize = 15)
    ax.set_ylim(-0.1,1.2)
    ax.set_xlabel("Time [s]", fontsize=15)
    ax.set_ylabel("Prn values", fontsize=15)
    ax.grid()
    for i,j in zip(centers, prn[0:N]):
        ax.text(i, 1.07, j, fontsize=14) 


def plot_prn_modified(N, prn):
    Rc = 1.023e06   # chip rate
    Tc = 1 / Rc     # chip period 
    
    t = np.arange(0, N*Tc, Tc)
         
    xticks = t
    yticks = [-1, 0, 1]
    centers = 0.5*(xticks[1:] + xticks[:-1]) 
    
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1,1,1)
    ax.step(t, prn[0:N], where='post', color='red', lw=3)
    ax.set_xticks(t)
    ax.set_yticks(yticks)
    title = "Prn to plot: " + str(prn[0:N])
    ax.set_title(title, fontsize=15)
    ax.set_ylim(-1.1,1.3)
    ax.set_xlabel("Time [s]", fontsize=15)
    ax.set_ylabel("Prn values", fontsize=15)
    ax.grid()
    for i,j in zip(centers, prn[0:N]):
        ax.text(i, 1.07, j, fontsize=14)

#Input: the data message (80 bits), the subcarrier (654720 samples) and the index of the PRN (4092 chips) to use,
#       that goes from 1 to 50 (while the prn index goes from 0 to 49) and a flag, to visualize it or not
#Output: the modulated signal
def boc(message, subcarrier, SV_index, flag):
    
    m = message #we use the variable m to avoid any modification of the original message
    
    #useful parameters
    Rb = 250   #The bit rate for GAL E1 is 250 symbols per second (equivalently, bits per second)
    Tb = 1 / Rb
    
    #first we convert the data message from bits to symbols
    for i in range(len(m)):
        if message[i] == 0:
            m[i] = 1
        elif message[i] == 1:
            m[i] = -1
    
    #we select the correct prn (based on the SV we have to communicate to)
    prn = e1bmodifiedcopy['Modified'][SV_index-1]
    
    #first the spreaded sequence is generated
    spreaded = np.zeros(327360)
    
    c = 0
    for i in range(len(m)):
        for j in range(len(prn)):
            spreaded[c] = m[i]*prn[j]
            c = c+1
    
    #so the spreaded sequence has been generated, now we apply the exact same procedure for the multiplication
    #by the subcarrier
    modulated = np.zeros(654720)

    c = 0
    for i in range(len(spreaded)):
        for j in range(2):
            modulated[c] = spreaded[i]*subcarrier[j]
            c = c+1

    #if the flag is set to true there is the visualization of the signals, otherwise no visualization
    #we plot the first 10 samples of the spreaded sequence 
    if(flag == True):
        fig = plt.figure(figsize=(15,5))
        fig.tight_layout()
        N = 10 
        t = np.arange(0,N*Tc, Tc)
        xticks = t
        
        #we plot the first sample of the data message (constant for the 10 represented chips)
        ax1 = fig.add_subplot(3,1,1)
        message_to_plot = np.tile(message[0], 10)
        ax1.step(t, message_to_plot, where='post', lw=4)
        ax1.set_xticks(xticks)
        ax1.set_yticks([-1,0,1])
        ax1.set_ylim(-1.8, 1.8)
        ax1.set_title("Spreaded sequence visualization", fontsize=25)
        ax1.set_xlabel("Time [s]", fontsize=15)
        ax1.set_ylabel("Message", fontsize=15)
        ax1.grid()

        #now we plot the first 10 values of the PRN
        ax2 = fig.add_subplot(3,1,2)
        prn_to_plot = prn[0:10]
        ax2.step(t, prn_to_plot, where='post', lw=4)
        ax2.set_xticks(xticks)
        ax2.set_ylim(-1.8, 1.8)
        ax2.set_yticks([-1,0,1])
        ax2.set_xlabel("Time [s]", fontsize=15)
        ax2.set_ylabel("PRN", fontsize=15)
        ax2.grid()

        #finally we plot the first 10 samples of the spreaded sequence, to verify that the multiplication was successful
        ax3 = fig.add_subplot(3,1,3)
        spreaded_to_plot = spreaded[0:10]
        ax3.step(t, spreaded_to_plot, where='post', lw=4)
        ax3.set_xticks(xticks)
        ax3.set_ylim(-1.8, 1.8)
        ax3.set_yticks([-1,0,1])
        ax3.set_xlabel("Time [s]", fontsize=15)
        ax3.set_ylabel("Spreaded", fontsize=15)
        ax3.grid()

        #now we create another figure for the modulated signal
        fig = plt.figure(figsize=(15,5))
        fig.tight_layout()
        
        ax1 = fig.add_subplot(3,1,1)
        N = 10 
        t = np.arange(0,N*Tc, Tc/2)
        t = t[:-1]
        
        #we start by plotting the spreaded sequence
        spreaded_to_plot = np.repeat(spreaded_to_plot[0:10], 2)
        xticks = t
        ax1.step(t, spreaded_to_plot[:-1], where='post', lw=4)
        ax1.set_xticks(xticks)
        ax1.set_yticks([-1,0,1])
        ax1.set_ylim(-1.8, 1.8)
        ax1.set_title("Modulated sequence visualization", fontsize=25)
        ax1.set_xlabel("Time [s]", fontsize=15)
        ax1.set_ylabel("Spreaded", fontsize=15)
        ax1.grid()

        #now we plot the subcarrier
        ax2 = fig.add_subplot(3,1,2)
        subcarrier_to_plot = subcarrier[0:19]
        ax2.step(t, subcarrier_to_plot, where='post', lw=4)
        ax2.set_xticks(xticks)
        ax2.set_ylim(-1.8, 1.8)
        ax2.set_yticks([-1,0,1])
        ax2.set_xlabel("Time [s]", fontsize=15)
        ax2.set_ylabel("Subcarrier", fontsize=15)
        ax2.grid()
        
        #finally we plot the first samples of the modulated signal
        ax3 = fig.add_subplot(3,1,3)
        modulated_to_plot = modulated[0:19]
        ax3.step(t, modulated_to_plot, where='post', lw=4)
        ax3.set_xticks(xticks)
        ax3.set_ylim(-1.8, 1.8)
        ax3.set_yticks([-1,0,1])
        ax3.set_xlabel("Time [s]", fontsize=15)
        ax3.set_ylabel("Modulated", fontsize=15)
        ax3.grid()        
        plt.show()
    return modulated

#This function simulates the additive white gaussian noise channel. It takes as input the signal without the noise,
#the power of the noise in dB and a flag that allows the generation of the noise only if it is true.
def awgn(s, noise_power_dB, flag):
    length = len(s)
    if flag == True:
        noise = 10 ** (noise_power_dB/20) * np.random.randn(length)  
        #this returns gaussian samples drawn from the standard normal distribution, so with zero mean and unitary
        #variance. The zero mean is okay, but the variance of the noise should be equal to its power, so we need
        #to multiply by the standard deviation of the noise (that is the sqrt of the power, so the sqrt of
        #(10**noise_power_dB/10). That is why we put the multiplication factor there.
        s = s + noise
    return s