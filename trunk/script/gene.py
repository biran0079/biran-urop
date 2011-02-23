#generate toy problem dataset
import sys,random,math
random.seed(None)
D=int(sys.argv[1])
N=int(sys.argv[2])
L=N
print "%d %d %d" %(D,N,L)

for i in range(D):
	x=[random.random() for j in range(N)]
	y=[]
	for i in x:
		if i>0.9:
			y.append(1)
		else:
			y.append(0);
	print " ".join([str(i) for i in x+y])
