
import java.io.File;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;


import edu.stanford.nlp.optimization.CGMinimizer;
import edu.stanford.nlp.optimization.DiffFunction;

public class CML extends MultiLabelClassifier {
	double[][] X, Y; // training dataset
	BitSet[] YB;
	double[] theta; // parameters

	int N, L, DistLabelNum;
	Map<BitSet, Integer> map = new HashMap<BitSet, Integer>();
	ArrayList<double[]> labelsetLst;

	// O(D*L)
	CML(double[][] X, double[][] Y) {
		this.X = X;
		this.Y = Y;
		N = X[0].length;
		L = Y[0].length;
		YB = new BitSet[Y.length];
		DistLabelNum = 0;
		labelsetLst = new ArrayList<double[]>();
		for (int i = 0; i < Y.length; i++) {
			YB[i] = toBitSet(Y[i]);
			if (!map.containsKey(YB[i])) {
				map.put(YB[i], DistLabelNum++);
				labelsetLst.add(Y[i]);
			}
		}
		theta = new double[N * L + (L - 1) * L * 2];
	}

	// O(G(LN+L^2D))
	public void train() {
		CGMinimizer opt = new CGMinimizer(false);
		NLikelihood func = new NLikelihood();

		// initialize(theta);
		long st = System.currentTimeMillis();
		theta = opt.minimize(func, 1.e-3, theta);
		long time_cost = System.currentTimeMillis() - st;
		System.out.println(time_cost / 1000.0 + " secs");
	}

	private void test(double[][] X, double[][] Y) {
		int correct = 0;
		for (int i = 0; i < X.length; i++) {
			if (predict(X[i], Y[i]))
				correct++;
		}
		System.out.printf("Accuracy: %f\n", correct * 1.0 / Y.length);
	}

	// O(DNL)
	private boolean predict(double[] x, double[] y) {
		double max_p = 0.0, p;
		double[] pred = null;
		double[] xw_buf = createXWBuf(theta, x);
		double[] gy_buf = createGYBuf(theta);

		for (int j = 0; j < labelsetLst.size(); j++) {
			p = P_Y(theta, x, j, xw_buf, gy_buf);
			if (p > max_p) {
				max_p = p;
				pred = labelsetLst.get(j);
			}
		}
		this.printPred(pred);
		return sameLabels(pred, y);
	}

	private BitSet toBitSet(double[] y) {
		BitSet res = new BitSet(y.length);
		for (int i = 0; i < y.length; i++)
			if (y[i] == 1.0)
				res.set(i);
		return res;
	}

	private double G(int j, double[] gy_buf) {
		return gy_buf[j];
	}

	private double[] createGYBuf(double[] theta) {
		double[] res = new double[labelsetLst.size()];
		for (int idx = 0; idx < labelsetLst.size(); idx++) {
			double[] y = labelsetLst.get(idx);
			int cur = N * L;
			for (int i = 0; i < L; i++)
				for (int j = i + 1; j < L; j++) {
					res[idx] += theta[cur + 2 * (int) y[i] + (int) y[j]];
					cur += 4;
				}
		}
		return res;
	}

	// O(L)
	private double F(int j, double[] xw_buf) {
		double res = 0.0;
		double[] y = labelsetLst.get(j);
		for (int i = 0; i < L; i++)
			if (y[i] == 1.0)
				res += xw_buf[i];
		return res;
	}

	private double P_Y(double[] theta, double[] x, int j, double[] xw_buf,
			double[] gy_buf) {
		return Math.exp(F(j, xw_buf) + G(j, gy_buf))
				/ Z(theta, x, xw_buf, gy_buf);
	}

	// O(N)
	private double xw(double[] theta, double[] x, int l) {
		double res = 0.0;
		for (int i = 0; i < N; i++)
			res += x[i] * theta[l * N + i];
		return res;
	}

	private double Z(double[] theta, double[] x, double[] xw_buf,
			double[] gy_buf) {
		double res = 0.0;
		for (int i = 0; i < labelsetLst.size(); i++) {
			res += Math.exp(F(i, xw_buf) + G(i, gy_buf));
		}
		return res;
	}

	// O(LN)
	double[] createXWBuf(double[] theta, double[] x) {
		double[] xw_buf = new double[L];
		for (int l = 0; l < L; l++)
			xw_buf[l] = xw(theta, x, l);
		return xw_buf;
	}

	private double P_Yl(double[] theta, double[] x, int l, double[] xw_buf,
			double[] gy_buf) {
		double res = 0.0;
		for (int i = 0; i < labelsetLst.size(); i++) {
			double[] y = labelsetLst.get(i);
			if (y[l] == 1.0) {
				res += Math.exp(F(i, xw_buf) + G(i, gy_buf));
			}
		}
		return res / Z(theta, x, xw_buf, gy_buf);
	}

	private double P_ijk(double[] theta, double[] x, int i, int j, int k,
			double[] xw_buf, double[] gy_buf) {
		double res = 0;
		for (int idx = 0; idx < labelsetLst.size(); idx++) {
			double[] y = labelsetLst.get(idx);
			if ((int) y[i] * 2 + (int) y[j] == k)
				res += Math.exp(F(idx, xw_buf) + G(idx, gy_buf));
		}
		return res / Z(theta, x, xw_buf, gy_buf);
	}

	/*
	 * 
	 * negative likelihood function class, which is to minimized
	 */
	class NLikelihood implements DiffFunction {

		class DerivativeAThread extends Thread {
			double[] x, y, res, xw_buf, gy_buf;
			final double[] theta;

			DerivativeAThread(double[] theta, double[] x, double[] y,
					double[] xw_buf, double[] gy_buf, double[] res) {
				this.x = x;
				this.y = y;
				this.theta = theta;
				this.xw_buf = xw_buf;
				this.res = res;
				this.gy_buf = gy_buf;
			}

			@Override
			public void run() {
				for (int l = 0; l < L; l++)
					for (int i = 0; i < N; i++)
						res[l * N + i] -= (y[l] - P_Yl(theta, x, l, xw_buf,
								gy_buf))
								* x[i];
			}
		}

		class DerivativeBThread extends Thread {
			double[] x, y, xw_buf, gy_buf, res;
			final double[] theta;

			DerivativeBThread(double[] theta, double[] x, double[] y,
					double[] xw_buf, double[] gy_buf, double[] res) {
				this.x = x;
				this.y = y;
				this.theta = theta;
				this.xw_buf = xw_buf;
				this.gy_buf = gy_buf;
				this.res = res;
			}

			@Override
			public void run() { 
				int t_idx = L * N;
				for (int i = 0; i < L; i++)
					for (int j = i + 1; j < L; j++) {
						for (int k = 0; k < 4; k++) {
							res[t_idx] -= ((int) y[i] * 2 + (int) y[j] == k ? 1.0
									: 0.0)
									- P_ijk(theta, x, i, j, k, xw_buf, gy_buf);
							t_idx++;
						}
					}
			}
		}

		@Override
		public double[] derivativeAt(double[] theta) {
			double[] res = new double[theta.length], x, y, xw_buf;
			double[][] tres = new double[X.length][theta.length];
			double[] gy_buf = createGYBuf(theta);
			
			DerivativeAThread[] at = new DerivativeAThread[X.length];
			DerivativeBThread[] bt = new DerivativeBThread[X.length];
			
			
			for (int idx = 0; idx < X.length; idx++) {
				x = X[idx];
				y = Y[idx];
				xw_buf = createXWBuf(theta, x);
				
				at[idx] = new DerivativeAThread(theta, x, y, xw_buf,gy_buf,
						tres[idx]);
				bt[idx] = new DerivativeBThread(theta, x, y, xw_buf,gy_buf,
						tres[idx]);

				at[idx].start();
				bt[idx].start();
			}

			for (int idx = 0; idx < X.length; idx++) {
				try {
					at[idx].join();
					bt[idx].join();
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			for (int idx = 0; idx < X.length; idx++) {
				for (int j = 0; j < res.length; j++)
					res[j] += tres[idx][j];
			}
			return res;
		}

		@Override
		public int domainDimension() {
			return theta.length;
		}

		// O(DL(D+N))
		@Override
		public double valueAt(double[] theta) {
			double res = 0.0;
			double[] gy_buf = createGYBuf(theta);

			for (int i = 0; i < X.length; i++) {
				double[] xw_buf = createXWBuf(theta, X[i]);
				int j = map.get(toBitSet(Y[i]));
				res -= F(j, xw_buf) + G(j, gy_buf)
						- Math.log(Z(theta, X[i], xw_buf, gy_buf));
			}
			return res; // negative likelihood
		}

	}

	public static void main(String[] args) {
		if (args.length != 2) {
			System.err.printf("Two parameters required:\n"
					+ "training data file name\n" + "testing data file name\n");
			return;
		}
		File train_file = new File(args[0]), test_file = new File(args[1]);

		double[][] train_x, test_x;
		double[][] train_y, test_y;
		Pair t = null;
		try {
			t = read_data(train_file);
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		train_x = t.first;
		train_y = t.second;
		CML crf = new CML(train_x, train_y);
		crf.train();
		try {
			t = read_data(test_file);
		} catch (Exception e) {
			e.printStackTrace();
		}
		test_x = t.first;
		test_y = t.second;

		crf.test(test_x, test_y);
	}

}
