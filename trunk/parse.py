#parse multi-label dataset into several singal label datasets
import sys
assert sys.argv.__len__()==2,"one parameter is required"
class_num=int(sys.argv[1])
line=raw_input();
a=line.split();
N=int(a[0])
D=int(a[1])
C=int(a[2])
print "%d %d" % (N,D)
for i in range(N):
	line=raw_input()
	s=line.split()
	for j in range(D):
		print "%s" % s[j],
	print "%s" % s[D+class_num-1]
