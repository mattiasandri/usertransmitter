import math
from bitarray import bitarray

#Function to append a string containing bits into a bitarray type ba
def appendOnBA(ba, string):
    for c in string:
        ba.append(int(c))


#function to code a literals string into a binary string using ASCII 7bit
def stringToASCII7(message):
    bin_message=""

    for char in message:
        word=format(ord(char),"b")
        word=word.zfill(7)
        bin_message=bin_message+word

    return bin_message


#This function will receive a string (message) of words and code it into a bitarray list
#syncPattern and SVID have to be strings
def createMessage(message, syncPattern, SVID):
    message=stringToASCII7(message)
    msgID=0
    messages=[]
    for i in range(math.floor(len(message)/30)):
        msg=bitarray()
        appendOnBA(msg, syncPattern)
        appendOnBA(msg, format(int(SVID), "b"))
        appendOnBA(msg, format(msgID, "b"))
        msgBody=message[msgID*30:((msgID*30)+29)]
        #if(len(msgBody)<30):

        appendOnBA(msg,msgBody)
        appendOnBA(msg,'000000')
        messages.append(msg)

    return messages


