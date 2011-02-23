import sys,random
random.seed()
l=[]
for line in sys.stdin:
	l.append(line)
random.shuffle(l)
for line in l:
	sys.stdout.write(line)
