def stringToASCII7(message):
    bin_message=""

    for char in message:
        word=format(ord(char),"b")
        word=word.zfill(7)
        bin_message=bin_message+word

    return bin_message
