#include<stdio.h>
#include<stdlib.h>
#include<time.h>

inline double rand_d(){
	//return random double value in [0,1]
	return 1.0*rand()/RAND_MAX;
}
double f(double a[],int n){
	//return 1.0 or 0.0
	double res=0.0;
	for(int i=0;i<n;i++)
		res+=a[i]*(i%4+1);
	return res>12 ? 1.0 : 0.0;
}
int main(int argc,char** args){
	if(argc!=3){
		printf("Two parameters required\n");
		return 1;
	}
	int N=atoi(args[1]),D=atoi(args[2]);
	double *x=new double[N];
	srand(time(0));

	printf("%d %d\n",N,D);

	for(int i=0;i<N;i++){
		for(int j=0;j<D;j++){
			x[j]=rand_d();
			printf("%lf ",x[j]);
		}
		printf("%lf\n",f(x,D));
	}
	delete[] x;
	return 0;
}
