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

	result = []
	
	# Exact Match Ratio
	ratio = 0
	hamming = 0
	acc=0
	for i in range(len(__predict)):
		__original[i].sort()
		__predict[i].sort()
		soi=set(__original[i])
		spi=set(__predict[i])
		hamming+=len(soi ^ spi)
		if len(soi | spi)==0:
			acc+=1.0
		else:
			acc+=float(len(soi & spi))/float(len(soi | spi))
		if(__original[i] == __predict[i]):
			ratio = ratio+1
	hamming=float(hamming)/(len(__labels)*len(__original))
	acc=acc/float(len(__original))
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
	print "Accuracy: %s" % acc
	print "Precision: %s" % P
	print "Recall: %s" % R
	print "Hamming loss: %s" % hamming
	print "F1-measure: %s" % result[1]


assert sys.argv.__len__()==2,"one argument required"
name=sys.argv[1]

labels=range(file(name+".test.pred").readline().split().__len__())

ans_pred=read_label(file(name+".test.pred"))

lr_pred=read_label(file(name+".test.lr.pred"))
print "logistic regression:"
measure(ans_pred,lr_pred,labels);
print

crf_pred=read_label(file(name+".test.crf.pred"))
print "conditional random field:"
measure(ans_pred,crf_pred,labels);
print

"""
crf_marginal_pred=read_label(file(name+".test.crf_marginal.pred"))
print "marginal conditional random field:"
measure(ans_pred,crf_marginal_pred,labels);
print


cml_pred=read_label(file(name+".test.cml.pred"))
print "CML:"
measure(ans_pred,cml_pred,labels);
"""


