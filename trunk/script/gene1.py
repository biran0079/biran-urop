import random

print 200,20,20
for i in range(40):
	x=[random.randint(0,1) for j in range(20)]
	y=x
	print " ".join([str(s) for s in x+y])
for i in range(40):
	x=[0]*10+[1]*10
	y=[1]*20
	print " ".join([str(s) for s in x+y])
for i in range(40):
	x=[1]*10+[0]*10
	y=[1]*20
	print " ".join([str(s) for s in x+y])
for i in range(40):
	x=[0]*20
	y=[0]*20
	print " ".join([str(s) for s in x+y])
for i in range(40):
	x=[1]*20
	y=[0]*20
	print " ".join([str(s) for s in x+y])

