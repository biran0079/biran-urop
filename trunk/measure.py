import sys
def read_label(file):
	res=[]
	for line in file:
		t=[]
		l=line.split()
		for j in range(l.__len__()):
			if(int(l[j])==1):
				t.append(j)
		res.append(t)
	return res

def measure(__original, __predict, __labels):
	"""
	Return exact match ratio, microaverage F-measure, and macroaverage F-measure.
	"""
	result = []
	
	# Exact Match Ratio
	ratio = 0
	for i in range(len(__predict)):
		__original[i].sort()
		__predict[i].sort()
		if(__original[i] == __predict[i]):
			ratio = ratio+1

	result.append(float(ratio)/len(__predict))
	
	# Microaverage and Macroaverage F-measure
	F = 0
	tp_sum = 0
	fp_sum = 0
	fn_sum = 0

	for j in __labels:
		tp = 0
		fp = 0
		fn = 0
		tn = 0

		for i in range(len(__predict)):
			if (j in __original[i] and j in __predict[i]):
				tp = tp + 1
			elif (j not in __original[i] and j in __predict[i]):
				fp = fp + 1
			elif (j in __original[i] and j not in __predict[i]):
				fn = fn + 1
			else:
				tn = tn + 1

		# 0/0 is treated as 0 and #labels does *not* reduced
		if (tp != 0 or fp != 0 or fn != 0):
			F = F+float(2*tp)/float(2*tp+fp+fn)

		tp_sum = tp_sum + tp
		fp_sum = fp_sum + fp
		fn_sum = fn_sum + fn

	P = float(tp_sum)/float(tp_sum+fp_sum)
	R = float(tp_sum)/float(tp_sum+fn_sum)

	result.append(2*P*R/(P+R))
	result.append(F/len(__labels))

	print "Exact match ratio: %s" % result[0]
	print "Microaverage F-measure: %s" % result[1]
	print "Macroaverage F-measure: %s" % result[2]


assert sys.argv.__len__()==2,"one argument required"
name=sys.argv[1]
lr=file(name+".test.lr.pred")
crf=file(name+".test.crf.pred")
crf_marginal=file(name+".test.crf_marginal.pred")
cml=file(name+".test.cml.pred")
ans=file(name+".test.pred")

ans_pred=read_label(ans)
lr_pred=read_label(lr)
crf_pred=read_label(crf)
crf_marginal_pred=read_label(crf_marginal)
cml_pred=read_label(cml)
labels=range(ans_pred[0].__len__())

for line in lr:
	lr_pred.append(line.split(" "));
for line in crf:
	crf_pred.append(line.split(" "));

print "logistic regression:"
measure(ans_pred,lr_pred,labels);
print
print "conditional random field:"
measure(ans_pred,crf_pred,labels);
print
print "marginal conditional random field:"
measure(ans_pred,crf_marginal_pred,labels);
print
print "CML:"
measure(ans_pred,cml_pred,labels);


