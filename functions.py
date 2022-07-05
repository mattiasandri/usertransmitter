import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from math import pi
import json
import csv
from bitarray import bitarray
from scipy import optimize


#THIS FILE CONTAINS THE LIST OF ALL THE FUNCTIONS USED FOR THE PROJECT, IN ORDER OF APPEARANCE IN THE MAIN FILE


#This function will create a message of ACK
#Receives the syncPattern, SVID as str and msgID as int
#Returns an array

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

    arr_msg = np.zeros(len(msg))

    for i in range(len(msg)):
        arr_msg[i] = int(msg.pop())
    arr_msg=np.flip(arr_msg)
    arr_msg=arr_msg.astype(np.int8)

    return arr_msg



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

    arr_msg = np.zeros(len(msg))
    
    for i in range(len(msg)):
        arr_msg[i] = msg.pop()
    arr_msg=np.flip(arr_msg)
    arr_msg=arr_msg.astype(np.int8)

    return arr_msg



#Function to append a string containing bits into a bitarray type ba

def appendOnBA(ba,object):
    if(isinstance(object,str)):
        for c in object: ba.append(int(c))
    else:
        for c in object: ba.append(c)



#Function that computes the CRC for the message.
#It receives a bitarray containing the information bits, in our case only
#the ACK or NACK strings
#returns a tuple containing the CRC calculated
#this function will be used by the createMessage*** functions
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

###################################################
# This part contains 4 functions to sum,          #
# multiply and divide binary vectors.             #
# They are needed for the computation of the CRC  #
###################################################

def BinPolyMul(b1,b2):
	result=[]
	for i in range(0,len(b1)+len(b2)-1):
		result.append(0)
	for i in range(0,len(b1)):
		if b1[i]==0:
			continue;
		for j in range(0,len(b2)):
			if b2[j]==0:
				continue
			sumDegree=i+j
			result[sumDegree]=1
	result=optiSizeVec(result)
	return result

def BinPolyDiv(b1,b2):
	quotient=[]
	for i in range(0,len(b1)):
		quotient.append(0)
	remainder=b1
	while (len(remainder)>=len(b2)):
		# print 'Iteration'
		dividendDegree=len(remainder)-1
		dividerDegree=len(b2)-1
		degreeDelta=dividendDegree-dividerDegree
		quotient[degreeDelta]=1
		subQuotient=[]
		for i in range(0,degreeDelta+1):
			if i==degreeDelta:
				subQuotient.append(1)
			else:
				subQuotient.append(0)
		summer=BinPolyMul(b2,subQuotient)
		remainder=BinPolySum(remainder,summer)
		remainder=optiSizeVec(remainder)
	quotient=optiSizeVec(quotient)
	remainder=optiSizeVec(remainder)
	return (quotient,remainder)

def BinPolySum(b1,b2):
	result=[]
	shortest=None
	longest=None
	if (len(b1)==len(b2)):
		for i in range(0,len(b1)):
			result.append((int(b1[i])+int(b2[i]))%2)
		result=optiSizeVec(result)
		return result
	elif (len(b1)>len(b2)):
		shortest=b2
		longest=b1
	else:
		shortest=b1
		longest=b2
	for i in range(0,len(shortest)):
		result.append((b1[i]+b2[i])%2)
	for j in range(len(shortest),len(longest)):
		result.append(longest[j])
	result=optiSizeVec(result)
	return result

def optiSizeVec(vec):
	optiVec=[]
	if len(vec)==0:
		return optiVec
	if len(vec)==1:
		if vec[0]==1:
			return vec
		else:
			return optiVec
	for i in range(1,len(vec)+1):
		if vec[-i]==0:
			continue
		else:
			for j in range(-len(vec),-i+1):
				optiVec.append(vec[j])
			break;
	return optiVec



# simple conversion from hexadecimal to binary

def hex2bin(h):
    integer = int(h, 16)
    binary = bin(integer)[2::]     #discarding the first two elements (i.e. 0b, not needed)
    binary = np.array([int(x) for x in binary])    #the binary representation is a numpy array of integers
    return binary


# simple conversion from 1 and 0 to -1 and 1

def conversion(prn):
    new_prn = np.zeros(len(prn))
    for i in range(len(prn)):
        if prn[i] == 0:
            new_prn[i] = 1
        elif prn[i] == 1:
            new_prn[i] = -1
    return np.array(new_prn, int)



#This function plots the first N chips of an original PRN (before the conversion), given a certain chip rate Rc

def plot_prn_zeros(N, prn, Rc):
    Tc = 1 / Rc     # chip period 
    
    t = np.arange(0, N*Tc, Tc)
    
    xticks = t
    yticks = [0, 1]
    centers = 0.5*(xticks[1:] + xticks[:-1]) 
    
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1,1,1)
    ax.step(t, prn[0:N], where='post', lw=3)
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


        
#This function plots the first N chips of a modified PRN (after the conversion), given a certain chip rate Rc

def plot_prn_modified(N, prn, Rc):
    Tc = 1 / Rc     # chip period 
    
    t = np.arange(0, N*Tc, Tc)
         
    xticks = t
    yticks = [-1, 0, 1]
    centers = 0.5*(xticks[1:] + xticks[:-1]) 
    
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1,1,1)
    ax.step(t, prn[0:N], where='post', lw=3)
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
        
        

#The function performs the BOC(1,1) modulation.
#The parameters are the data message (80 bits), the subcarrier (654720 samples, it can also be created inside the function but #it's more efficient to create it outside only once and to pass it as a parameter, since it doesn't change), the symbol rate of #the message, the chip rate of the PRN, the array of PRNs, the index of the PRN to use, the number of prn to use for a single #symbol of the message and a flag (to visualize graphically the signals or not).
#The output is the modulated signal

def boc(message, subcarrier, Rb, Rc, array, SV_index, prn_x_symbol, flag):   
    m = np.copy(message)     #we use the variable m to avoid any modification of the original message
    
    Tb = 1 / Rb     #bit length
    Tc = 1 / Rc     #chip length
    
    #first we convert the data message from bits to symbols
    for i in range(len(m)):
        if m[i] == 0:
            m[i] = 1
        elif m[i] == 1:
            m[i] = -1
    
    #we select the correct prn (based on the SV we have to communicate to; SV goes from 1 to 50, while PRNs from 0 to 49)
    prn = array[SV_index-1]
    
    #first the spreaded sequence is generated
    len_spreaded = len(message)*prn_x_symbol*len(prn)
    spreaded = np.zeros(len_spreaded)
    
    c = 0
    for i in range(len(m)):
        for j in range(len(prn)):
            spreaded[c] = m[i]*prn[j]
            c = c+1
    
    #so the spreaded sequence has been generated, now we apply the exact same procedure for the multiplication
    #by the subcarrier
    
    modulated = np.zeros(len(subcarrier))

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
        message_to_plot = np.tile(m[0], 10)
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



#Function that samples a signal with a certain sampling frequency Fs. The number of the packet is the "index" variable
#If flag is true, it also prints the modulated signals, the original signals and the corresponding time vectors.

def sampling(signal, Fs, symbol_duration, total_time_vector, index, flag):
    #As already said, each symbol of the modulated signal has a duration of Tc / 2. Therefore if we use Ts = Tc / 4 we
    #just need to repeat each symbol two times. This can be easily done with the function np.repeat()

    Ts = 1 / Fs      #sampling period, equal to Tc/4 (in general it should be Tc/2, Tc/4, Tc/6 and so on)

    #creation of the sampled signal
    repetitions = symbol_duration / Ts
    final_length = int(len(signal)*repetitions)
    sampled_signal = np.repeat(signal, repetitions)

    #creation of the time vector
    begin = index*final_length
    t_sampled = total_time_vector[begin:begin+final_length]   #1309440 values spaced apart by Tc / 4 seconds each
    
    if (flag==True):
        print("Before sampling:", signal[0:9])
        print("After sampling: ", sampled_signal[0:18])
        print("Length after sampling:", len(sampled_signal))
        print("\nTime vector:", t_sampled)
        print("Time vector length:", len(t_sampled))

    return (sampled_signal, t_sampled)



#This function increases the amplitude of a given signal multiplying it by the square root of the power to set
#If flag = True, the signal is plotted. Index is the number of the packet. It returns the amplified signal

def hpa(signal, time_vector, power_to_set, index, flag):
    power = np.sum(signal**2)/len(signal)
    
    #to set a different power P we just need to multiply the signal by the sqrt(P)
    amplified_signal = np.sqrt(power_to_set)*signal
    new_power = np.mean(amplified_signal**2)   #equivalent way to compute the power of a signal
    
    #plot of the signal (first N samples)
    if(flag==True):
        N = 20
        xticks = time_vector[0:N-1]
        yticks = [-np.sqrt(power_to_set), 0, np.sqrt(power_to_set)]
        fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(1,1,1)
        ax.step(time_vector[0:N], amplified_signal[0:N], where='post', lw=3)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        title = "Amplified signal of packet " + str(index+1) + " at the transmitter"
        ax.set_title(title, fontsize=15)
        ax.set_xlabel('Time [s]', fontsize=15)
        ax.set_ylabel('Amplitude', fontsize=15)
        ax.grid()
    
    return amplified_signal



#simple function to convert from linear to dB

def lin2dB(x):
    return 10*np.log10(x)



#simple function to convert from dB to linear

def dB2lin(x):
    return 10**(x/10)



#Function that interpolates the Free Space Path Loss of the orbit propagator file
#Parameters: original FSPL samples, original sampling period, starting point in seconds of the interpolated samples
#to be returned, number of interpolated samples needed, interpolated sampling period
#returns: interpolated FSPL

def GetFSPL(originalFSPL, originalSamplingPeriod, startingTime, numInterpSamples, interpSamplingPeriod, plotFlag=False):

    SampleTime = (np.asarray(range(0,len(originalFSPL))))*originalSamplingPeriod #create the x vector (original)
    InterpSampleTime = (np.asarray(range(0,numInterpSamples)))*interpSamplingPeriod #create the x vector (interpolated)
    InterpSampleTime = np.asarray([i+startingTime for i in InterpSampleTime]) #shifts it 

    popt, pcov = optimize.curve_fit(linf, SampleTime, originalFSPL, p0=[-0.5, 5.60238e+09], full_output=False)
    interpFSPL = linf(InterpSampleTime, popt[0], popt[1])

    if(plotFlag):
        plt.figure(figsize=(10, 6))
        plt.plot(SampleTime,originalFSPL,label="Original FSPL",linewidth=6)
        plt.plot(InterpSampleTime,interpFSPL,label="Interpolated FSPL",linewidth=8)
        plt.legend(loc="upper right")

    return  interpFSPL



#Function for the interpolation of free space path loss 

def linf(x, a, b):
    return np.asarray((a*x)+b)



#Function that returns the amplitude of the received signal, corresponding to a packet number 'index'.
#Takes as input the array of the FSPL  (after the interpolation, with a number of samples equal to the
#number of packets times the final length), the index (number of the packet), the final length of the signal corresponding to #the returned time vector, and all the fixed parameters for the link budget

def return_amplitude(array, index, Pt, Gt, Gr, final_length):
    FSPL = array[index*final_length]  
    Pr_dB = lin2dB(Pt) + Gt + Gr - FSPL
    Pr = dB2lin(Pr_dB)
    return np.sqrt(Pr)



#Function that, given the total time vector and the final length of the signal of one packet, returns the time vector of the #packet number "index" (first packet has index 0)

def return_time_vector(total_time_vector, index, final_length):
    begin = final_length*index
    time = total_time_vector[begin:begin+final_length]
    return time



#Function that interpolates the doppler shifts of the orbit propagator file
#Parameters: original Doppler samples, original sampling period, starting point in seconds of the interpolated samples
#to be returned, number of interpolated samples needed, interpolated sampling period
#Returns: interpolated doppler shifts

def GetDopplerShift(originalDopplerSamples, originalSamplingPeriod, startingTime, numInterpSamples, interpSamplingPeriod, plotFlag=False):

    SampleTime = (np.asarray(range(0,len(originalDopplerSamples))))*originalSamplingPeriod #create the x vector (original)
    InterpSampleTime = (np.asarray(range(0,numInterpSamples)))*interpSamplingPeriod #create the x vector (interpolated)
    InterpSampleTime = np.asarray([i+startingTime for i in InterpSampleTime]) #shifts it 

    popt, pcov = optimize.curve_fit(cosf, SampleTime, originalDopplerSamples, p0=[3.45376, 0.001], full_output=False)
    interpDopplerSamples = cosf(InterpSampleTime, popt[0], popt[1])

    if(plotFlag):
        plt.figure(figsize=(10, 6))
        plt.plot(SampleTime,originalDopplerSamples,label="Original Doppler Shift",linewidth=6)
        plt.plot(InterpSampleTime,interpDopplerSamples,label="Interpolated Doppler Shift",linewidth=8)
        plt.legend(loc="upper right")

    return interpDopplerSamples



#function for the interpolation of doppler frequencies
def cosf(x, A, nu):
    return A * np.cos(nu * x) # sine function with amplitude A and angular frequency



#Function that returns the doppler shifts for the packet number "index". It takes in input the complete array
#of Doppler frequencies for the all packets

def return_dopplers(array, index, final_length):
    begin = final_length*index
    return array[begin:begin+final_length]



#Function that computes the I/Q samples of a signal, given the doppler frequencies and the time vector of the signal
#If flag is true, it plots them

def IQ_samples(signal, time, dopplers, flag):
    I = signal*np.cos(2*pi*dopplers*time)
    Q = signal*np.sin(2*pi*dopplers*time)
    
    if (flag==True):
        N = 20
        fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(1,1,1)
        xticks = time[0:N]
        ax.stem(time[0:N], I[0:N])
        ax.set_xticks(xticks)
        plt.xticks(rotation=45)
        ax.set_title("I samples before noise addition", fontsize=15)
        ax.set_xlabel("Time [s]", fontsize=15)
        ax.set_ylabel("I samples", fontsize=15)
        ax.grid()
        
        fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(1,1,1)
        xticks = time[0:N]
        ax.stem(time[0:N], Q[0:N])
        ax.set_xticks(xticks)
        plt.xticks(rotation=45)
        ax.set_title("Q samples before noise addition", fontsize=15)
        ax.set_xlabel("Time [s]", fontsize=15)
        ax.set_ylabel("Q samples", fontsize=15)
        ax.grid()
        
    return (I, Q)



#This function simulates the additive white gaussian noise. It takes as input the signal without the noise,
#the power of the noise in dB and a flag that allows the generation of the noise only if it is true.

def awgn(s, noise_power_dB, flag):
    length = len(s)
    if flag == True:
        noise = (10 ** (noise_power_dB/20)) * np.random.randn(length)  
        #np.random.randn(length) returns gaussian samples drawn from the standard normal distribution, so with zero mean and             #unitary variance. The zero mean is okay, but the variance of the noise should be equal to its power, so we need
        #to multiply by the standard deviation of the noise (that is the sqrt of the power, so the sqrt of
        #(10**noise_power_dB/10). That is why we put the multiplication factor there.
        s = s + noise
    return s



#Function that quantizes an array by converting it to a float of "nbits" bits

def quantizationFloat(array, nbits):
    if(nbits==16):
        q_array=array.astype(np.float16)
    elif(nbits==32):
        q_array=array.astype(np.float32)
    elif(nbits==64):
        q_array=array.astype(np.float64)
    elif(nbits==128):
        q_array=array.astype(np.float128)
    else:
        raise ValueError('nbits must be 16, 32, 64 or 128')
            
    return q_array



#Function to write float16 I Q samples into a file

def writeFileFloat(filename, I_samples, Q_samples):
    file=open(filename, 'wb')
    for j in range(len(I_samples)):
        file.write(I_samples[j])
        file.write(Q_samples[j])
    file.close()
    

    
#This function will read nsamples from a .bin file containing the signal, with samples in the
#floating point notation of length nbits, and return the I_samples and Q_samples vector

def readQuantizedFloat(filename,nbits,nsamples):
    if(nbits==16):
        message=np.fromfile(filename,  dtype=np.float16)
    elif(nbits==32):
        message=np.fromfile(filename,  dtype=np.float32)
    elif(nbits==64):
        message=np.fromfile(filename,  dtype=np.float64)
    elif(nbits==128):
        message=np.fromfile(filename,  dtype=np.float128)
    else:
        raise ValueError('nbits must be 16, 32, 64 or 128')
    
    I_samples = np.array(message[0],ndmin=1)
    Q_samples = np.array(message[1],ndmin=1)

    for i in range(2, nsamples):
        if i % 2: Q_samples=np.append(Q_samples,message[i])
        else : I_samples=np.append(I_samples,message[i])
    
    return I_samples,Q_samples
    
    
    
#This function performs an alternative quantization. First normalizing the values between 0
#and 2^nbits, and after that performing quantization of the values.
#IMPORTANT: if the array contains values smaller than lower_bound (higher than upper_bound)
#they will be "clipped" to the lower_bound (upper_bound).
#If bounds are not passed as argouments, they are choosen finding the max/min
#value of the array

def quantizationBounds(array,nbits,lower_bound=None,upper_bound=None):
    if(upper_bound==None): upper_bound=np.max(array)
    if(lower_bound==None): lower_bound=np.min(array)
    q_array=np.clip(array,lower_bound,upper_bound)
    q_array = q_array-lower_bound
    array_norm=np.linalg.norm(q_array)
    q_array=(q_array/array_norm)*(2**nbits)
    q_array = q_array.astype(np.uint16)
    return q_array   



#Function to write I and Q samples already quantized into uint16 into a binary file.

def writeFileBin(filename, I_samples, Q_samples):
    file=open('signal.bin', 'wb')
    for j in range(len(I_samples)):
        file.write((int(I_samples[j])).to_bytes(2, byteorder='big', signed=False))
        file.write((int(Q_samples[j])).to_bytes(2, byteorder='big', signed=False))
    file.close()
    
    
    
#Function to write I and Q samples (given as uint16) into a text file as bit strings,
#separated with a space.

def writeFileChar(filename, I_samples, Q_samples):
    file=open('signal.txt', 'w')
    for j in range(len(I_samples)):
        file.write(format(I_samples[j], 'b').zfill(16))
        file.write(' ')
        file.write(format(Q_samples[j], 'b').zfill(16))
        file.write(' ')
    file.close()
    

#function to code a literals string into a binary string using ASCII 7bit
#useless for the user transmitter, but might be useful somehow

def stringToASCII7(message):
    bin_message=""

    for char in message:
        word=format(ord(char),"b")
        word=word.zfill(7)
        bin_message=bin_message+word

    return bin_message