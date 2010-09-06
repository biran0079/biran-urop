import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;


public class MultiLabelClassifier {
	protected void printPred(double[] pred){
		for(int i=0;i<pred.length;i++){
			if(i!=0)System.out.print(' ');
			System.out.print(pred[i]==1.0?"1":"0");
		}
		System.out.println();
	}
	protected double vectorMul(double[] a,double[] b){
		double res=0;
		for(int i=0;i<a.length;i++)
			res+=a[i]*b[i];
		return res;
	}

	protected boolean sameLabels(double[] y1, double[] y2) {
		int len = y1.length;

		for (int i = 0; i < len; i++)
			if (y1[i] != y2[i])
				return false;
		return true;
	}
	protected void initialize(double[] w){
		for(int i=0;i<w.length;i++)
			w[i]=Math.random()*0.001;
	}
	public static class Pair {
		double[][] first;
		double[][] second;

		Pair(double[][] x, double[][] y) {
			this.first = x;
			this.second = y;
		}
	}
	
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
	
}
