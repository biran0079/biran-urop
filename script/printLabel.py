import sys
f=file(sys.argv[1])
line=f.readline().split()
D=int(line[0])
N=int(line[1])
L=int(line[2])
for line in f:
	l=line.split()
	for j in range(N,N+L):
		if int(l[j])==1:
			print j-N,
	print
