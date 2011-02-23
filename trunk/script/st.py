import sys
f=file(sys.argv[1])
s=f.readline().split()
N=int(s[1])
L=int(s[2])
ct=[0]*L
for line in f:
	l=line.split()
	for i in range(N,N+L):
		ct[i-N]+=int(l[i])
for i in ct:
	print i
