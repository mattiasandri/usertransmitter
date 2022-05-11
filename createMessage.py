import math
from bitarray import bitarray
from crc import *

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
