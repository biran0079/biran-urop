#combine predictions made by binary logistic regression together
import sys
assert sys.argv.__len__()==2,"one argument required"
name=sys.argv[1]
ss=file(name+".test").readline().split()
D = int(ss[0])
L = int(ss[2])
files=[]
for i in range(L):
	files.append(file(name+".test.pred."+str(i+1)));
for i in range(D):
	for file in files:
		v=float(file.readline().split()[0])
		if v>=0.5:
			print 1,
		else:
			print 0,
	print
