import java.io.File;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;

import edu.stanford.nlp.optimization.CGMinimizer;
import edu.stanford.nlp.optimization.DiffFunction;

public class CRFMultiThreadsWithRegularization extends MultiLabelClassifier {
	double[][] X, Y; // training dataset
	BitSet[] YB;
	double[] theta;

	int N, L, DistLabelNum;
	Map<BitSet, Integer> map = new HashMap<BitSet, Integer>();
	ArrayList<double[]> labelsetLst;

	// O(D*L)
	CRFMultiThreadsWithRegularization(double[][] X, double[][] Y) {
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

		theta = new double[N * L + DistLabelNum];
	}

	// O(GDL(Ld+N))
	public void train() {
		CGMinimizer opt = new CGMinimizer(false);
		NLikelihood func = new NLikelihood();

		// initialize(theta);
		long st = System.currentTimeMillis();
		theta = opt.minimize(func, 1e-3, theta);
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
		double[] pred1 = null, pred2, pred;
		double[] xw_buf = createXWBuf(theta, x);

		for (int j = 0; j < labelsetLst.size(); j++) {
			p = P_Y(theta, x, labelsetLst.get(j), xw_buf);
			if (p > max_p) {
				max_p = p;
				pred1 = labelsetLst.get(j);
			}
		}

		pred2 = new double[L];
		for (int l = 0; l < L; l++) {
			if (xw_buf[l] >= 0) {
				pred2[l] = 1.0;
			}
		}

		if (max_p > P_Y(theta, x, pred2, xw_buf)) {
			// pred 1 is better
			pred = pred1;
		} else {
			// pred 2 is better
			pred = pred2;
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

	/**
	 * O(1) return the index of the label combination. If this y does not appear
	 * in training set, then return -1;
	 * */
	private int getIdx(double[] y) {
		Integer res = map.get(toBitSet(y));
		return res == null ? -1 : res;
	}

	// O(L)
	private double G(double[] theta, double[] y) {
		int j = getIdx(y);
		return j == -1 ? 0 : theta[L * N + j];// getB_j(theta,j)
	}

	// O(L)
	private double F(double[] y, double[] xw_buf) {
		double res = 0.0;
		for (int i = 0; i < L; i++)
			if (y[i] == 1.0)
				res += xw_buf[i];
		return res;
	}

	/**
	 * O(NL)
	 * 
	 * @return P(y|x)
	 * */
	private double P_Y(double[] theta, double[] x, double[] y, double[] xw_buf) {
		return Math.exp(F(y, xw_buf) + G(theta, y)) / Z(theta, x, xw_buf);
	}

	// O(N)
	private double xw(double[] theta, double[] x, int l) {
		double res = 0.0;
		for (int i = 0; i < N; i++)
			res += x[i] * theta[l * N + i];
		return res;
	}

	// O(Ld)
	private double Z(double[] theta, double[] x, double[] xw_buf) {
		double res = 1.0;
		double[] y;

		for (int l = 0; l < L; l++)
			res *= 1 + Math.exp(xw_buf[l]);

		for (int j = 0; j < labelsetLst.size(); j++) {
			y = labelsetLst.get(j);
			res += Math.exp(F(y, xw_buf)) * (Math.exp(theta[L * N + j]) - 1);
		}
		return res;
	}

	/**
	 * O(Ld)
	 * 
	 * @return P(Y_l==1 | x)
	 * */
	private double P_Yl(double[] theta, double[] x, int l, double[] xw_buf) {
		double up = Math.exp(xw_buf[l]),down=1.0,tmp;

		for (int tl = 0; tl < L; tl++) {
			tmp= 1 + Math.exp(xw_buf[tl]);
			if (tl != l)
				up *= tmp;
			down*=tmp;
		}
		double[] y;

		for (int j = 0; j < labelsetLst.size(); j++) {
			y = labelsetLst.get(j);
			tmp=Math.exp(F(y, xw_buf)) * (Math.exp(theta[N * L + j]) - 1);
			if (y[l] != 0.0)
				up += tmp;
			down+=tmp;
		}
		return up / down;
	}

	// O(LN)
	double[] createXWBuf(double[] theta, double[] x) {
		double[] xw_buf = new double[L];
		for (int l = 0; l < L; l++)
			xw_buf[l] = xw(theta, x, l);
		return xw_buf;
	}

	/*
	 * 
	 * negative likelihood function class, which is to minimized
	 */
	class NLikelihood implements DiffFunction {

		final static int core_number = 2;
		double lambda = 0.5; // regularization

		class DerivativeAThread extends Thread {
			double[] x, y, xw_buf, res;
			final double[] theta;

			DerivativeAThread(double[] theta, double[] x, double[] y,
					double[] xw_buf, double[] res) {
				this.x = x;
				this.y = y;
				this.theta = theta;
				this.xw_buf = xw_buf;
				this.res = res;
			}
			
			//O(L(Ld+N))
			@Override
			public void run() {
				for (int l = 0; l < L; l++) {
					double temp = (y[l] == 1.0 ? 1.0 : 0.0)
							- P_Yl(theta, x, l, xw_buf);
					for (int i = 0; i < N; i++) {
						double delta = x[i] * temp;
						res[l * N + i] = delta;
					}
				}
			}
		}

		class DerivativeBThread extends Thread {
			double[] x, y, xw_buf, res;
			final double[] theta;

			DerivativeBThread(double[] theta, double[] x, double[] y,
					double[] xw_buf, double[] res) {
				this.x = x;
				this.y = y;
				this.theta = theta;
				this.xw_buf = xw_buf;
				this.res = res;
			}

			//O(Ld)
			@Override
			public void run() {
				for (int j = 0; j < labelsetLst.size(); j++) {
					double[] ty = labelsetLst.get(j);
					double delta = (sameLabels(y, ty) ? 1.0 : 0.0)
							- P_Y(theta, x, ty, xw_buf);
					res[L * N + j] = delta;
				}
			}
		}

		// O(DL(Ld+N))
		@Override
		public double[] derivativeAt(double[] theta){
			double[] res = new double[theta.length];
			double[][] tres = new double[core_number][theta.length];

			DerivativeAThread[] at = new DerivativeAThread[core_number];
			DerivativeBThread[] bt = new DerivativeBThread[core_number];

			for (int idx = 0; idx < X.length; idx += core_number) {
				for (int i = 0; i < core_number && idx + i < X.length; i++) {
					double[] x = X[idx + i], y = Y[idx + i];
					double[] xw_buf = createXWBuf(theta, x);
					at[i] = new DerivativeAThread(theta, x, y, xw_buf, tres[i]);
					bt[i] = new DerivativeBThread(theta, x, y, xw_buf, tres[i]);
					at[i].start();
					bt[i].start();
				}
				for (int i = 0; i < core_number && idx + i < X.length; i++) {
					try {
						at[i].join();
						bt[i].join();
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}

					for (int j = 0; j < theta.length; j++) {
						res[j] += tres[i][j];
					}
				}
			}

			for (int j = 0; j < theta.length; j++) {
				res[j] = -(res[j] - lambda * theta[j]);
			}
			return res;
		}

		@Override
		public int domainDimension() {
			return theta.length;
		}

		// O(DL(N+d))
		@Override
		public double valueAt(double[] theta) {
			double res = 0.0;
			for (int i = 0; i < X.length; i++) {
				double[] xw_buf = createXWBuf(theta, X[i]);

				res += F(Y[i], xw_buf) + G(theta, Y[i])
						- Math.log(Z(theta, X[i], xw_buf));
			}
			double penalty = 0.0;
			for (int i = 0; i < theta.length; i++) {
				penalty += theta[i] * theta[i];
			}
			penalty *= lambda / 2.0;
			res -= penalty;
			return -res; // negative likelihood
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
		CRFMultiThreadsWithRegularization crf = new CRFMultiThreadsWithRegularization(
				train_x, train_y);
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