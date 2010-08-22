#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<assert.h>
#include<vector>
using namespace std;
typedef vector<double> VD;
class Instance{
	public:
		VD x;
		double y;
};
typedef vector<Instance> Dataset;


inline double abs(double a){return a>0?a:-a;}
inline double rand_d(){
	//return random double value in [0,1]
	return 1.0*rand()/RAND_MAX;
}
double operator*(const VD& a,const VD& b){
	assert(a.size()==b.size());
	double res=0;
	int len=a.size();
	for(int i=0;i<len;i++)
		res+=a[i]*b[i];
	return res;
}

double P(const VD& x,const VD& w){
	// Para x: input feature vector
	// Para w: parameter in logistic regression moddel
	// x[0] is always 1.0 
	// return: P(Y=1 | X, W)
	return 1.0/(1.0+exp(x*w));
}
void initialize(VD& w){
	for(int i=0;i<w.size();i++)
		w[i]=rand_d()*0.001;
}
double l(const Dataset& data,const VD& w){
	// log likelihood
	// return P(Y|X,W)
	double res=0.0;
	for(int i=0;i<data.size();i++){
		const VD &x=data[i].x;
		double y=data[i].y,pxw=P(x,w);
		res+= y==1.0 ? log(pxw+1e-8):log(1-pxw+1e-8);	
		// to avoid cases like "log(0)".
	}
	return res;
}

double length(const VD& a){
	double res=0;
	for(int i=0;i<a.size();i++)res+=a[i]*a[i];
	return sqrt(res);
}
VD train(const Dataset& data){
	// Train Logistic Regression using gradient ascet
	// Para data: training dataset
	// return: parameter w in Logistic Regression model
	assert(data.size()>0);
	// Assumming traing dataset is nunempty
	int N=data[0].x.size();//dimension of feature vector
	VD w(N),delta(N),w2d(N),whd(N),wd(N);
	// Parameter vector should has the 
	// same length as instances in dataset
	initialize(w);
	// Fill in w with some small random values
	double ita=0.01;	//learning rate
	const double epi=0.01;	//threshold for termination of gradient ascent
	double prev_l=-1e100,cur_l;
	int times=1;
	const int max_it=200;
	const double min_delta=0.1;
	while(1){
		for(int i=0;i<N;i++)delta[i]=0.0;
		for(int j=0;j<data.size();j++){
			const VD &x=data[j].x;
			double y=data[j].y,pxw=P(x,w);
			for(int i=0;i<N;i++){
				delta[i]+=x[i]*(pxw-y);
			}
		}
		/*
		for(int i=0;i<N;i++){
			wd[i]=w[i]-ita*delta[i];
			w2d[i]=w[i]-2*ita*delta[i];
			whd[i]=w[i]-0.5*ita*delta[i];
		}
		double ld=l(data,wd),l2d=l(data,w2d),lhd=l(data,whd);
		if(ld > l2d && ld > lhd){
			for(int i=0;i<N;i++)w[i]=wd[i];
			cur_l=ld;
		}else if(l2d > ld && l2d > lhd){
			for(int i=0;i<N;i++)w[i]=w2d[i];
			cur_l=l2d;
			ita*=2.0;
			printf("twice~\n");
		}else{
			for(int i=0;i<N;i++)w[i]=whd[i];
			cur_l=lhd;
			ita*=0.5;
			printf("half~\n");
		}
		*/
		for(int i=0;i<N;i++)w[i]=w[i]+ita*delta[i];
		cur_l=l(data,w);
		/*
		   printf("delta length in iteration %d: %lf\n",times,length(delta));
		   if(length(delta) < min_delta)break;
		   */
		if(prev_l > cur_l){
			// close to local min, reduce step size
			ita*=0.95;	
		}
		//printf("Likelihood in iteration %d: %lf\n",times,cur_l);
		if(abs(cur_l-prev_l) < epi) break;
		prev_l=cur_l;

		if(times>=max_it)break;
		times++;
	}
	return w;
}
void test(const Dataset& data,const VD& w){
	int correct=0;
	double pred;
	for(int i=0;i<data.size();i++){
		pred=P(data[i].x,w)>0.5 ? 1.0 : 0.0;
		correct+= pred==data[i].y ? 1 : 0;
		printf("%d\n",(int)pred);
	}
	printf("Accuracy:\n%.2lf%%\n",100.0*correct/data.size());
}
Dataset read_data(FILE* file){
	// read dataset from a text file.
	// File Format:
	// Line 1: two integers N,D 
	// 	which are number of instances and dimension of feature vector
	// Line 2-N+1 : D+1 double values
	// 	first D values are features, 
	// 	the last value is the label, which is either 0.0 or 1.0
	assert(file);
	int N,D;
	Dataset res;
	Instance ins;
	double t;
	fscanf(file,"%d%d",&N,&D);
	for(int i=0;i<N;i++){
		ins.x.clear();
		ins.x.push_back(1.0);	//x[0] is always 1.0
		for(int j=0;j<D;j++){
			fscanf(file,"%lf",&t);
			ins.x.push_back(t);
		}
		fscanf(file,"%lf",&t);
		assert(t==0.0 || t==1.0);
		ins.y=t;		//the label
		res.push_back(ins);
	}
	return res;
}
int main(int argc,char** args){
	if(argc!=3){
		printf("Two parameters required:\n\
				training data file name\n\
				testing data file name\n");
		return -1;
	}
	//printf("reading data from files...\n");
	FILE *train_file=fopen(args[1],"r"),*test_file=fopen(args[2],"r");

	Dataset train_set=read_data(train_file),
		test_set=read_data(test_file);

	fclose(train_file);
	fclose(test_file);

	//printf("start training...\n");

	VD w=train(train_set);

	//printf("start testing...\n");
	test(test_set,w);
	return 0;
}
