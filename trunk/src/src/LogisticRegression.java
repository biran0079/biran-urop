import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.InputStream;
import java.util.Random;

import edu.stanford.nlp.optimization.CGMinimizer;
import edu.stanford.nlp.optimization.DiffFunction;


public class LogisticRegression {
	
	private static double vectorMul(double[] a,double[] b) throws Exception{
		if(a.length!=b.length)
			throw new Exception("vector length not identical");
		double res=0;
		for(int i=0;i<a.length;i++)
			res+=a[i]*b[i];
		return res;
	}

	/**
	 * @param  x: input feature vector. x[0] is always 1.0 
	 * @param  w: parameter in logistic regression model
	 * @return P(Y=1 | X, W)
	 * @throws Exception 
	 * */
	public static double P(double[] x,double[] w){
		try {
			return 1.0/(1.0+Math.exp(vectorMul(x,w)));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return 0;
	}
	private static void initialize(double[] w){
		for(int i=0;i<w.length;i++)
			w[i]=Math.random()*0.001;
	}
	
	static class NLikelihood implements DiffFunction{
		double[][] x;
		double[] y;
		
		NLikelihood(double[][] x,double[] y){
			this.x=x;
			this.y=y;
		}
		
		@Override
		public double[] derivativeAt(double[] w) {
			int N=x[0].length;
			double[] delta=new double[N];
			for(int i=0;i<N;i++)delta[i]=0.0;
			for(int j=0;j<x.length;j++){
				double pxw=P(x[j],w);
				for(int i=0;i<N;i++){
					delta[i]-=x[j][i]*(pxw-y[j]);
				}
			}
			return delta;
		}

		@Override
		public int domainDimension() {
			return x[0].length;
		}

		/**
		 * log likelihood
		 * @param  x: features of dataset
		 * @param  y: labels of dataset
		 * @param  w: parameter in logistic regression model
		 * @return P(Y|X,W)
		 * @throws Exception 
		 * */

		@Override
		public double valueAt(double[] w) {
			double res=0.0,pxw;
			for(int i=0;i<x.length;i++){
				pxw=P(x[i],w);
				res+= y[i]==1.0 ? Math.log(pxw+1e-8):Math.log(1-pxw+1e-8);	
				// to avoid cases like "log(0)".
			}
			return -res;
		}
		
	}
	/**
	 * Train Logistic Regression using gradient ascent
	 * @param  x: features of dataset
	 * @param  y: labels of dataset
	 * @return parameter in logistic regression model
	 * @throws Exception 
	 * */
	static double[] train(double[][] x,double[] y) throws Exception{
		if(x.length==0)
			throw new Exception("cannot train from empty dataset");

		int N=x[0].length;		//dimension of feature vector
		double[] w=new double[N];
		CGMinimizer opt=new CGMinimizer();
		initialize(w);
		return opt.minimize(new NLikelihood(x,y), 1e-3, w);
	}
	
	/**
	 * Train Logistic Regression using gradient ascent
	 * @param  x: features of testing dataset
	 * @param  y: labels of testing dataset
	 * @param  w: parameter in logistic regression model
	 * */
	static void test(double[][] x,double[] y,double[] w){
		int correct=0;
		double pred;
		for(int i=0;i<x.length;i++){
			pred=P(x[i],w)>0.5 ? 1.0 : 0.0;
			correct+= pred==y[i] ? 1 : 0;
			System.out.printf("%f\n",P(x[i],w));
		}
		System.out.printf("Accuracy:\n%.2f%%\n",100.0*correct/x.length);
	}
	
	static class Pair{
		double[][] first;
		double[] second;
		Pair(double[][] x,double[] y){
			this.first=x;
			this.second=y;
		}
	};
	/** read dataset from a text file.
	 * File Format:
	 *	Line 1: two integers N,D 
	 *		which are number of instances and dimension of feature vector
	 *	Line 2-N+1 : D+1 double values
	 *		first D values are features, 
	 *		the last value is the label, which is either 0.0 or 1.0
	 * 
	 * @param file: the dataset file
	 * @throws Exception 
	 */
	static Pair read_data(File file) throws Exception{
		int N,D;
		
		BufferedReader in=new BufferedReader(new FileReader(file));
		
		String[] ss=in.readLine().split(" ");
		N=Integer.valueOf(ss[0]);
		D=Integer.valueOf(ss[1]);
		double[][]x=new double[N][D+1];
		double[] y=new double[N];
		for(int i=0;i<N;i++){
			ss=in.readLine().split(" ");
			for(int j=0;j<D;j++){
				x[i][j]=Double.valueOf(ss[j]);
			}
			x[i][D]=1.0;
			y[i]=Double.valueOf(ss[D]);
			if(y[i]!=0.0 && y[i]!=1.0)
				throw new Exception("invalid label value: only 0.0 and 1.0 is allowed");
		}
		in.close();
		return new Pair(x,y);
	}
	public static void main(String[] args) throws Exception{
		if(args.length!=2){
			System.err.printf("Two parameters required:\n"
					+"training data file name\n"
					+"testing data file name\n");
			return;
		}
		File train_file=new File(args[0]),test_file=new File(args[1]);

		double[][] train_x,test_x;
		double[] train_y,test_y;
		Pair t;
		
		t=read_data(train_file);
		train_x=t.first;
		train_y=t.second;

		t=read_data(test_file);
		test_x=t.first;
		test_y=t.second;



		double[] w=train(train_x,train_y);

		test(test_x,test_y,w);
	}

}
