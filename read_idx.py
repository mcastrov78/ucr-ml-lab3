import sys
import struct

"""
0x08: unsigned byte
0x09: signed byte
0x0B: short (2 bytes)
0x0C: int (4 bytes)
0x0D: float (4 bytes)
0x0E: double (8 bytes) 
"""

def fromubytes(bytes):
    return int.from_bytes(bytes, byteorder="big", signed=False)

def fromsbytes(bytes):
    return int.from_bytes(bytes, byteorder="big", signed=True)
    
def fromfbytes(bytes):
    return struct.unpack('f', bytes)

dtypesn = {8: 1, 9: 1, 11: 2, 12: 4, 13: 4, 14: 8}
dtypesconv = {8: fromubytes, 9: fromsbytes, 11: fromsbytes, 12: fromsbytes, 13: fromfbytes, 14: fromfbytes}

def read(fname,maxn=None):
    f = open(fname, "rb")
    magic = f.read(4)
    dtype = magic[2]
    ndims = magic[3]
    dims = []
    count = 1
    for i in range(ndims):
        n = fromsbytes(f.read(4))
        if maxn is not None and n > maxn:
            n = maxn
        dims.append(n)
        count *= n

    res = []
    for i in range(count):
        n = f.read(dtypesn[dtype])
        res.append(dtypesconv[dtype](n))
    f.close()
    
    return (res,dims)

main = read

if __name__ == "__main__":
    main(sys.argv[1])