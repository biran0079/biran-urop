import sys
assert len(sys.argv)==2,"one argument required"
f=file(sys.argv[1])
line=f.readline()
s=line.split()
D=int(s[0])
N=int(s[1])
L=int(s[2])
for line in f:
	s=line.split()
	print " ".join(s[N:])
