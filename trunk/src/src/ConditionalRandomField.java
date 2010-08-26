import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;


import edu.stanford.nlp.optimization.CGMinimizer;
import edu.stanford.nlp.optimization.DiffFunction;
import edu.stanford.nlp.optimization.QNMinimizer;
import edu.stanford.nlp.optimization.SGDMinimizer;
import edu.stanford.nlp.optimization.ScaledSGDMinimizer;

public class ConditionalRandomField {
	double[][] X,Y;		//	training dataset
	double[] theta;		//	parameters
	int N,L,DistLabelNum;
	Map<BitSet,Integer> map=new HashMap<BitSet,Integer>();

	
	ConditionalRandomField(double[][] X,double[][] Y){
		this.X=X;
		this.Y=Y;
		N=X[0].length;
		L=Y[0].length;
		DistLabelNum=0;
		BitSet tb;
		for(int i=0;i<Y.length;i++){
			tb=toBitSet(Y[i]);
			if(!map.containsKey(tb)){
				map.put(tb, DistLabelNum++);
			}
		}
		theta=new double[N*L+DistLabelNum];
	}
	private static void initialize(double[] w){
		for(int i=0;i<w.length;i++)
			w[i]=Math.random()*0.001;
	}
	
	public void train(){
		System.out.println("start training...");
		CGMinimizer opt=new CGMinimizer(false);
		NLikelihood func=new NLikelihood();

		initialize(theta);
		
		theta=opt.minimize(func, 1.e-2, theta,10);
		
		
		System.out.println("training finished");
	}
	
	private void test(double[][] X, double[][] Y) {
		int correct=0;
		for(int i=0;i<X.length;i++){
			if(predict(X[i],Y[i]))
				correct++;
		}
		System.out.printf("Accuracy: %f\n",correct*1.0/Y.length);
	}
	
	private boolean predict(double[] x,double[] y) {
		double max_p=0.0,p;
		BitSet pred1 = null,pred2,pred;

		/*
		double[] ans = null;
		pred=new BitSet(L);
		for(int i=0;i<(1<<L);i++){
			for(int j=0;j<L;j++)
				if((i & (1<<j))!=0){
					pred.set(j,true);
				}else{
					pred.set(j,false);
				}
			double[] ty=toArray(pred);
			p=P_Y(theta,x,ty);
			System.out.printf("%d %f\n",i,p);
			if(p>max_p){
				max_p=p;
				
				ans=ty;
			}
		}
		System.out.println(max_p);
		*/
		
		for(BitSet b:map.keySet()){
			p=P_Y(theta,x,toArray(b));
			//System.out.println(p);
			if(p>max_p){
				max_p=p;
				pred1=b;
			}
		}
		pred2=new BitSet(L);
		for(int l=0;l<L;l++){
			if(xw(theta,x,l)>=0){
				pred2.set(l);
			}
		}
		if(max_p > P_Y(theta,x,toArray(pred2))){
			//	pred 1 is better
			pred=pred1;
			for(int i=0;i<L;i++)
				System.out.print(pred1.get(i)?"1 ":"0 ");
			System.out.println();
		}else{
			//	pred 2 is better
			pred=pred2;
			for(int i=0;i<L;i++)
				System.out.print(pred2.get(i)?"1 ":"0 ");
			System.out.println();
		}
		
		
		return pred.equals(toBitSet(y));
	}
	private double[] toArray(BitSet b) {
		double[] res=new double[L];
		for(int i=0;i<L;i++)
			res[i]=b.get(i)?1.0:0.0;
		return res;
	}
	private BitSet toBitSet(double[] y){
		BitSet res=new BitSet(y.length);
		for(int i=0;i<y.length;i++)
			if(y[i]==1.0)
				res.set(i);
		return res;
	}
	
	
	public static class Pair {
		double[][] first;
		double[][] second;

		Pair(double[][] x, double[][] y) {
			this.first = x;
			this.second = y;
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
	
	private double getA_li(double[] theta,int l,int i){
		return theta[l*N+i];
	}
	private double getB_j(double[] theta,int j){
		return theta[L*N+j];
	}
	
	private void increaseA_li(double[] theta,int l,int i,double v){
		theta[l*N+i]+=v;
	}
	private void increaseB_j(double[] theta,int j,double v){
		theta[L*N+j]+=v;
	}
	
	private double F(double[] theta,double[] x,double[] y){
		double res=0.0;
		for(int l=0;l<L;l++)
			if(y[l]==1.0){
				res+=xw(theta,x,l);
			}
		return res;
	}
	private double G(double[] theta,double[] y){
		int j=getIdx(y);
		//System.out.println(j);
		return j==-1?0:getB_j(theta,j);
	}
	/**
	 * @return P(y|x)
	 * */
	private double P_Y(double[] theta, double[] x,double[] y) {
		return Math.exp(F(theta,x,y)+G(theta,y))/Z(theta,x);
	}
	private double xw(double[] theta,double[] x,int l){
		double res=0.0;
		for(int i=0;i<N;i++)
			res+=x[i]*getA_li(theta,l,i);
		return res;
	}
	
	/*
	 * 	return the index of the label combination.
	 *  If this y does not appear in training set, then return -1;
	 * */
	private int getIdx(double[] y) {
		Integer res=map.get(toBitSet(y));
		return res==null?-1:res;
	}
	
	//	correct
	private double Z(double[] theta,double[] x){
		double res=1.0;
		for(int l=0;l<L;l++)
			res*=1+Math.exp(xw(theta,x,l));
/*
		double t=0.0;
		double[] y=new double[L];
		for(int i=0;i<(1<<L);i++){
			for(int j=0;j<L;j++)
				if((i & (1<<j))!=0){
					y[j]=1.0;
				}else{
					y[j]=0.0;
				}
			t+=Math.exp(F(theta,x,y)+G(theta,y));
		}
*/
		for(BitSet b:map.keySet()){
			double[] y=toArray(b);
			double tf=F(theta,x,y);
			res+=Math.exp(tf+G(theta,y));
			res-=Math.exp(tf);
		}
		return res;
	}
	
	/**
	 * correct
	 * @return P(Y_l==1 | x)
	 * */
	private double P_Yl(double[] theta, double[] x, int l) {
		double res=Math.exp(xw(theta,x,l));
		for(int tl=0;tl<L;tl++)
			if(tl!=l)
				res*=1+Math.exp(xw(theta,x,tl));
		for(BitSet b:map.keySet()){
			if(b.get(l)==false)continue;
			double[] y=toArray(b);
			double tf=F(theta,x,y);
			res+=Math.exp(tf+G(theta,y));
			res-=Math.exp(tf);
		}
		res/=Z(theta,x);
		
		/*
		double t_res=0;
		double[] ty=new double[L];
		for(int i=0;i<(1<<L);i++)
			if((i & (1<<l))!=0){
				for(int j=0;j<L;j++)
					if((i & (1<<j))!=0)
						ty[j]=1.0;
					else
						ty[j]=0.0;
				t_res+=P_Y(theta,x,ty);
			}
		System.out.println(res+" "+t_res);
		*/
		
		return res;
	}
	
	private boolean sameLabels(double[] y1, double[] y2) {
		int len=y1.length;
		if(len!=y2.length)
			try {
				throw new Exception("tow label sets not compatible");
			} catch (Exception e) {
				e.printStackTrace();
			}
		for(int i=0;i<len;i++)
			if(y1[i]!=y2[i])
				return false;
		return true;
	}
	
	
	/*
	 * 	negative likelihood function class, which is to minimized
	 * */
	class NLikelihood implements DiffFunction{
		
		@Override
		public double[] derivativeAt(double[] theta) {
			double[] res=new double[theta.length];
			
			for(int idx=0;idx<X.length;idx++){
				double[] x=X[idx],y=Y[idx];
				for(int l=0;l<L;l++){
					for(int i=0;i<N;i++)
						increaseA_li(res, l, i, x[i]*(y[l]==1.0?1.0:0.0 - P_Yl(theta,x,l)));
				}
				for(BitSet b:map.keySet()){
					int j=map.get(b);
					increaseB_j(res, j, sameLabels(y,toArray(b))?1.0:0.0 - P_Y(theta,x,toArray(b)));
				}
			}
			for(int i=0;i<res.length;i++)
				res[i]=-res[i];			//negative likelihood, so negate derivative 

			
			
			double[] t=theta.clone(),t_res=new double[theta.length];
			double delta=1e-3;
			for(int i=0;i<theta.length;i++){
				t[i]+=delta;
				t_res[i]=-(valueAt(t)-valueAt(theta))/delta;
				t[i]-=delta;
			}

			System.out.println();
			printArray(t_res);
			printArray(res);
			
			return res;
		}
		
		private void printArray(double[] res) {
			for(int i=0;i<res.length;i++)
				System.out.print(res[i]+" ");
			System.out.println();
		}

		@Override
		public int domainDimension() {
			return N*L+DistLabelNum;
		}

		@Override
		public double valueAt(double[] theta) {
			double res=0.0;
			for(int i=0;i<X.length;i++){
				res+=F(theta,X[i],Y[i])+G(theta,Y[i])-Math.log(Z(theta,X[i]));
			}
			//System.out.println(-res);
			return -res;	//negative likelihood
		}
		
	}
	public static void main(String[] args){
		if(args.length!=2){
			System.err.printf("Two parameters required:\n"
					+"training data file name\n"
					+"testing data file name\n");
			return;
		}
		File train_file=new File(args[0]),test_file=new File(args[1]);

		double[][] train_x,test_x;
		double[][] train_y,test_y;
		Pair t = null;
		try {
			t=read_data(train_file);
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		train_x=t.first;
		train_y=t.second;
		ConditionalRandomField crf=new ConditionalRandomField(train_x,train_y);
		crf.train();
		try {
			t=read_data(test_file);
		} catch (Exception e) {
			e.printStackTrace();
		}
		test_x=t.first;
		test_y=t.second;

		crf.test(test_x,test_y);
	}

}
