#extract features and run logistic regression
import sys,os
assert(sys.argv.__len__()==2),"one parameter is required."
name=sys.argv[1]
train_file=file(name+".train","r")
test_file=file(name+".test","r")
C=int(train_file.readline().split()[2])
for i in range(1,C+1):
	in_file_name=name+".train"
	train_file_name=in_file_name+"."+str(i)
	pred_file_name=name+".test.pred"+"."+str(i)
	cmd="parse.py %d < %s > %s" % (i,in_file_name,train_file_name)
	os.system(cmd)
	in_file_name=name+".test"
	test_file_name=in_file_name+"."+str(i)
	cmd="parse.py %d < %s > %s" % (i,in_file_name,test_file_name)
	os.system(cmd)
	cmd="java -cp .\;.\stanford-classifier.jar LogisticRegression %s %s > %s" % (train_file_name,test_file_name,pred_file_name)
	os.system(cmd)
train_file.close()
test_file.close()
