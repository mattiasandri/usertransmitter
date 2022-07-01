import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import math

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