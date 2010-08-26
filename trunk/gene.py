#generate toy problem dataset
import sys,random
assert sys.argv.__len__()==4,"three arguments required"
D=int(sys.argv[1])
N=int(sys.argv[2])
L=int(sys.argv[3])
print "%d %d %d" %(D,N,L)
for i in range(D):
	x=[]
	for j in range(N):
		x.append(random.random())
	for j in range(L):
		if x[j%N]>0.5 :
			x.append( 1.0 )
		else:
			x.append( 0.0 )
	for t in x:
		print t,
	print
