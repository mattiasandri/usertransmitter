#This function will receive a string (message) and code it into a bitarray list


def createMessage(message, syncPattern, SVID):
    message=
    msgID=0
    messages=[]
    for i in range(math.floor(len(message)/30)):
        msg=bitarray()
        msg.append(syncPattern)
        msg.append(SVID)
        msg.append(format(msgID, "b"))
        msg.append(message[msgID*30,((msgID*30)+29)])