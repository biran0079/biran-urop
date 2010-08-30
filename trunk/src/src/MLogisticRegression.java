import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import edu.stanford.nlp.optimization.CGMinimizer;
import edu.stanford.nlp.optimization.DiffFunction;


public class MLogisticRegression {
	double[][] X,Y;		//	training dataset
	double[][] w;		//	parameters
	int N,L;
	
	MLogisticRegression(double[][] X,double[][] Y){
		this.X=X;
		this.Y=Y;
		N=X[0].length;
		L=Y[0].length;
		w=new double[L][N];
	}
	private double vectorMul(double[] a,double[] b){
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
	public double P(double[] x,double[] w){
		try {
			return 1.0/(1.0+Math.exp(vectorMul(x,w)));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return 0;
	}
	private void initialize(double[] w){
		for(int i=0;i<w.length;i++)
			w[i]=Math.random()*0.001;
	}
	
	class NLikelihood implements DiffFunction{
		int idx;	//label index
		NLikelihood(int label_idx){
			this.idx=label_idx;
		}
		@Override
		public double[] derivativeAt(double[] w) {
			int N=X[0].length;
			double[] delta=new double[N];
			for(int i=0;i<N;i++)delta[i]=0.0;
			for(int j=0;j<X.length;j++){
				double pxw=P(X[j],w);
				for(int i=0;i<N;i++){
					delta[i]-=X[j][i]*(pxw-Y[j][idx]);
				}
			}
			return delta;
		}

		@Override
		public int domainDimension() {
			return N;
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
			for(int i=0;i<X.length;i++){
				pxw=P(X[i],w);
				res+= Y[i][idx]==1.0 ? Math.log(pxw+1e-8):Math.log(1-pxw+1e-8);	
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
	void train() throws Exception{
		CGMinimizer opt=new CGMinimizer();
		//initialize(w);
		for(int l=0;l<L;l++)
			w[l]=opt.minimize(new NLikelihood(l), 1e-3, w[l]);
	}
	
	private void test(double[][] X, double[][] Y) {
		int correct=0;
		for(int i=0;i<X.length;i++){
			if(predict(X[i],Y[i]))
				correct++;
		}
		System.out.printf("Accuracy: %f\n",correct*1.0/Y.length);
	}
	
	private boolean sameLabels(double[] y1, double[] y2) {
		int len=y1.length;
		for(int i=0;i<len;i++)
			if(y1[i]!=y2[i])
				return false;
		return true;
	}
	
	private boolean predict(double[] x,double[] y) {
		double[] pred;
		pred=new double[L];
		for(int l=0;l<L;l++){
			if(vectorMul(x,w[l])<0.0)
				pred[l]=1.0;
			else
				pred[l]=0.0;
		}
		for(int i=0;i<L;i++){
			if(i!=0)System.out.print(' ');
			System.out.print(pred[i]==1.0?"1":"0");
		}
		System.out.println();
		return sameLabels(pred,y);
	}
	
	static class Pair{
		double[][] first;
		double[][] second;
		Pair(double[][] x,double[][] y){
			this.first=x;
			this.second=y;
		}
	};
	/**
	 * read dataset from a text file. File Format: Line 1: two integers D,N,L
	 * which are number of instances in dataset, dimension of feature vector,
	 * and number of features. Line 2-N+1 : N+L double values first N values are
	 * features, the last L values are the label, which are either 0.0 or 1.0
	 * 
	 * @param file
	 *            : the dataset file
	 * @throws Exception
	 */
	public static Pair read_data(File file) throws Exception {
		
		int N, D, L;

		BufferedReader in = new BufferedReader(new FileReader(file));

		String[] ss = in.readLine().split(" ");
		D = Integer.valueOf(ss[0]); // # of instances
		N = Integer.valueOf(ss[1]); // # of features in each instance
		L = Integer.valueOf(ss[2]); // # if labels
		double[][] x = new double[D][N + 1];
		double[][] y = new double[D][L];
		for (int i = 0; i < D; i++) {
			ss = in.readLine().split(" ");
			for (int j = 0; j < N; j++) {
				x[i][j] = Double.valueOf(ss[j]);
			}
			x[i][N] = 1.0;
			for (int j = N; j < N + L; j++) {
				y[i][j - N] = Double.valueOf(ss[j]);
				if (y[i][j - N] != 0.0 && y[i][j - N] != 1.0)
					throw new Exception(
							"invalid label value: only 0.0 and 1.0 is allowed");
			}
		}
		in.close();
		return new Pair(x, y);
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
		double[][] train_y,test_y;
		Pair t;
		
		t=read_data(train_file);
		train_x=t.first;
		train_y=t.second;

		t=read_data(test_file);
		test_x=t.first;
		test_y=t.second;
		MLogisticRegression lr=new MLogisticRegression(train_x,train_y);
		lr.train();
		lr.test(test_x,test_y);
	}

}
