package sparse;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;

import sparse.Instance.Entry;

import edu.stanford.nlp.optimization.CGMinimizer;
import edu.stanford.nlp.optimization.DiffFunction;

public class MCML {

	double[] w = null; // parameters
	double lambda = 1e-2, bias = 1;
	boolean debug = false;
	double tolerance = 1e-3;

	File paramFile = null;
	File outputParamFile = null;
	int L, N;

	Map<Set<Integer>, Integer> S;

	class NegativeLogLikelihood implements DiffFunction {
		LogLikelihood l;

		NegativeLogLikelihood(LogLikelihood l) {
			this.l = l;
		}

		@Override
		public double valueAt(double[] _w) {
			return -l.valueAt(_w);
		}

		@Override
		public int domainDimension() {
			return l.domainDimension();
		}

		@Override
		public double[] derivativeAt(double[] _w) {
			double[] res = l.derivativeAt(_w);
			for (int i = 0; i < res.length; i++)
				res[i] = -res[i];
			return res;
		}

	};

	double[] getP_l(double[] xwBuf, double[] gyBuf, double z) {
		double[] res = new double[L];
		double[] e = new double[L];
		double temp = 1.0;
		for (int i = 0; i < L; i++) {
			e[i] = Math.exp(xwBuf[i]);
			temp *= (1 + e[i]);
		}
		for (int i = 0; i < L; i++)
			res[i] = temp * e[i] / (1 + e[i]);
		for (Map.Entry<Set<Integer>, Integer> entry : S.entrySet()) {
			Set<Integer> y = entry.getKey();
			int idx = entry.getValue();
			for (Integer i : y)
				res[i] += Math.exp(F(y, xwBuf) + gyBuf[idx])
						- Math.exp(F(y, xwBuf));
		}
		for (int i = 0; i < L; i++)
			res[i] /= z;
		return res;
	}

	double[][][] getP_i_j_k(double[] xwBuf, double[] gyBuf) {
		double[][][] P_i_j_k = new double[L][L][4];

		for (Map.Entry<Set<Integer>, Integer> entry : S.entrySet()) {
			Set<Integer> y = entry.getKey();
			int idx = entry.getValue();
			int[] lb = new int[L];
			for (Integer i : y)
				lb[i] = 1;
			for (int i = 0; i < L; i++)
				for (int j = i + 1; j < L; j++)
					P_i_j_k[i][j][lb[i] * 2 + lb[j]] += Math.exp(F(y, xwBuf)
							+ gyBuf[idx]);
		}
		double z = Z(xwBuf, gyBuf);
		for (int i = 0; i < L; i++)
			for (int j = i + 1; j < L; j++) {
				for (int k = 0; k < 4; k++)
					P_i_j_k[i][j][k] /= z;
			}
		return P_i_j_k;
	}

	Queue<Job> toDo = new ConcurrentLinkedQueue<Job>(); // thread-safe
	Queue<Job> done = new ConcurrentLinkedQueue<Job>();

	class Job {
		double[] _w;
		Instance ins;
		double[] res;
		double[] gyBuf;

		Job(double[] _w, double[] gyBuf, Instance ins) {
			this._w = _w;
			this.gyBuf = gyBuf;
			this.ins = ins;
		}

		void doWork() {
			res = new double[_w.length];
			int[] lb = new int[ins.L];
			for (Integer i : ins.y)
				lb[i] = 1;
			double[] xwBuf = createXwBuf(_w, ins);
			double z = Z(xwBuf, gyBuf);
			double[] P_Y_l = getP_l(xwBuf, gyBuf, z);
			for (int l = 0; l < ins.L; l++) {
				double temp = (lb[l] == 1 ? 1.0 : 0.0) - P_Y_l[l];
				for (Entry e : ins.x)
					res[l * ins.N + e.idx] += temp * e.val;
			}
			int cur = L * N;

			double[][][] P_i_j_k = getP_i_j_k(xwBuf, gyBuf);

			for (int i = 0; i < L; i++)
				for (int j = i + 1; j < L; j++)
					for (int k = 0; k < 4; k++) {
						res[cur] += (lb[i] * 2 + lb[j] == k ? 1.0 : 0.0)
								- P_i_j_k[i][j][k];

						cur++;
					}
		}
	}

	class Worker extends Thread {
		@Override
		public void run() {
			while (true) {
				Job j = toDo.poll();
				if (j == null)
					break;
				j.doWork();
				done.add(j);
			}
		}
	}

	class LogLikelihood implements DiffFunction {
		Dataset train; // training set

		LogLikelihood(Dataset train) {
			this.train = train;
		}

		@Override
		public double valueAt(double[] _w) {
			double res = 0;
			double[] gyBuf = createGYBuf(_w);
			for (int i = 0; i < train.D; i++) {
				Instance ins = train.data.get(i);
				double[] xwBuf = createXwBuf(_w, ins);
				double lnZ = Math.log(Z(xwBuf, gyBuf));
				res += F(ins.y, xwBuf) + gyBuf[S.get(ins.y)] - lnZ;
			}
			double panelty = 0;
			for (int i = 0; i < w.length; i++)
				panelty += _w[i] * _w[i];
			panelty *= lambda / 2.0;
			res -= panelty;
			return res;
		}

		@Override
		public int domainDimension() {
			return w.length;
		}

		@Override
		public double[] derivativeAt(double[] _w) {
			double[] res = new double[_w.length];
			double[] gyBuf = createGYBuf(_w);
			toDo.clear();
			done.clear();
			for (int idx = 0; idx < train.D; idx++) {
				Instance ins = train.data.get(idx);
				toDo.add(new Job(_w, gyBuf, ins));
			}
			int coreNumber = Runtime.getRuntime().availableProcessors();
			Worker[] workers = new Worker[coreNumber];
			for (int i = 0; i < coreNumber; i++) {
				workers[i] = new Worker();
				workers[i].start();
			}
			for (int i = 0; i < coreNumber; i++) {
				try {
					workers[i].join();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			while (!done.isEmpty()) {
				Job j = done.poll();
				for (int i = 0; i < res.length; i++)
					res[i] += j.res[i];
			}

			for (int i = 0; i < res.length; i++) {
				res[i] -= _w[i] * lambda;
			}
			return res;
		}
	}

	double P_Y_l(double[] _w, int l, double[] xwBuf, double[] gyBuf, double z) {
		double res = Math.exp(xwBuf[l]) / (1 + Math.exp(xwBuf[l]));
		for (int i = 0; i < L; i++)
			res *= (1 + Math.exp(xwBuf[i]));
		for (Map.Entry<Set<Integer>, Integer> entry : S.entrySet()) {
			Set<Integer> y = entry.getKey();
			if (!y.contains(l))
				continue;
			int idx = entry.getValue();
			res += Math.exp(F(y, xwBuf) + gyBuf[idx]) - Math.exp(F(y, xwBuf));
		}
		res /= z;
		return res;
	}

	double P_Y_i_j_k(double[] _w, int I, int J, int K, double[] xwBuf,
			double[] gyBuf, double z) {
		double res = 1.0 / (1 + Math.exp(xwBuf[I])) / (1 + Math.exp(xwBuf[J]));
		if (K / 2 != 0)
			res *= Math.exp(xwBuf[I]);
		if (K % 2 != 0)
			res *= Math.exp(xwBuf[J]);
		for (int i = 0; i < L; i++)
			res *= (1 + Math.exp(xwBuf[i]));
		for (Map.Entry<Set<Integer>, Integer> entry : S.entrySet()) {
			Set<Integer> y = entry.getKey();
			int[] lb = new int[L];
			for (Integer i : y)
				lb[i] = 1;
			if (lb[I] != K / 2 || lb[J] != K % 2)
				continue;
			int idx = entry.getValue();
			res += Math.exp(F(y, xwBuf) + gyBuf[idx]) - Math.exp(F(y, xwBuf));
		}
		res /= z;
		// System.out.println(res);
		return res;
	}

	double P_Y_l(double[] _w, int l, double[] xwBuf, double[] gyBuf) {
		return P_Y_l(_w, l, xwBuf, gyBuf, Z(xwBuf, gyBuf));
	}

	double F(Set<Integer> y, double[] xw_buf) {
		double res = 0.0;
		for (Integer i : y)
			res += xw_buf[i];
		return res;
	}

	int getIdx(Set<Integer> y) {
		Integer res = S.get(y);
		return res == null ? -1 : res;
	}

	double P_Y(Set<Integer> y, double[] xw_buf, double[] gy_buf, double z) {
		int j = getIdx(y);
		if (j != -1)
			return Math.exp(F(y, xw_buf) + gy_buf[j]) / z;
		else
			return Math.exp(F(y, xw_buf)) / z;
	}

	double P_Y(Set<Integer> y, double[] xw_buf, double[] gy_buf) {
		return P_Y(y, xw_buf, gy_buf, Z(xw_buf, gy_buf));
	}

	double Z(double[] xwBuf, double[] gyBuf) {
		double res = 1.0;
		for (int i = 0; i < L; i++)
			res *= (1 + Math.exp(xwBuf[i]));
		for (Map.Entry<Set<Integer>, Integer> entry : S.entrySet()) {
			Set<Integer> y = entry.getKey();
			int idx = entry.getValue();
			res += Math.exp(F(y, xwBuf) + gyBuf[idx]) - Math.exp(F(y, xwBuf));
		}
		return res;
	}

	// O(G(LN+L^2D))
	public void train(Dataset trainSet) throws IOException {
		this.L = trainSet.L;
		this.N = trainSet.N;
		trainSet.setBias(bias);
		S = new HashMap<Set<Integer>, Integer>();
		for (int i = 0; i < trainSet.D; i++) {
			Instance ins = trainSet.data.get(i);
			if (!S.containsKey(ins.y)) {
				S.put(ins.y, S.size());
			}
		}
		if (paramFile != null)
			w = initializeParam(paramFile);
		else
			w = new double[trainSet.N * trainSet.L + L * (L - 1) * 2];

		CGMinimizer optimizer = new CGMinimizer(!debug);
		DiffFunction f = new NegativeLogLikelihood(new LogLikelihood(trainSet));
		w = optimizer.minimize(f, tolerance, w);
		
		if (outputParamFile != null)
			this.printParam(outputParamFile);
	}

	Result test(Dataset testSet) {
		testSet.setBias(bias);
		ArrayList<Set<Integer>> pred = new ArrayList<Set<Integer>>();
		for (int i = 0; i < testSet.D; i++)
			pred.add(predict(testSet.data.get(i)));
		return new Result(testSet, pred);
	}

	Set<Integer> predict(Instance ins) {
		double max_p = 0.0, p;
		Set<Integer> pred = new HashSet<Integer>();
		double[] xw_buf = createXwBuf(w, ins);
		double[] gy_buf = createGYBuf(w);
		double z = Z(xw_buf, gy_buf);

		for (int i = 0; i < L; i++)
			if (xw_buf[i] > 0)
				pred.add(i);
		max_p = P_Y(pred, xw_buf, gy_buf);
		for (Map.Entry<Set<Integer>, Integer> entry : S.entrySet()) {
			Set<Integer> y = entry.getKey();
			p = P_Y(y, xw_buf, gy_buf, z);
			if (p > max_p) {
				max_p = p;
				pred = y;
			}
		}
		return pred;
	}

	double[] createGYBuf(double[] theta) {
		double[] res = new double[S.size()];
		for (Map.Entry<Set<Integer>, Integer> entry : S.entrySet()) {
			Set<Integer> y = entry.getKey();
			int idx = entry.getValue();
			int cur = L * N;
			int[] lb = new int[L];
			for (Integer i : y)
				lb[i] = 1;
			for (int i = 0; i < L; i++)
				for (int j = i + 1; j < L; j++) {
					res[idx] += theta[cur + lb[i] * 2 + lb[j]];
					cur += 4;
				}
		}
		return res;
	}

	double[] createXwBuf(double[] _w, Instance ins) {
		double[] xwBuf = new double[ins.L];
		for (int i = 0; i < ins.L; i++)
			xwBuf[i] = ins.dotProdX(_w, i * ins.N);
		return xwBuf;
	}

	void selectBestParameter(Dataset trainSet) throws IOException {
		int trainset_size = trainSet.D * 7 / 10;
		Collections.shuffle(trainSet.data);
		Dataset trainSub = new Dataset(trainSet, 0, trainset_size), validationSet = new Dataset(
				trainSet, trainset_size, trainSet.D);

		double[] lambda_cand = { 1e-4, 1e-3, 1e-2, 0.1, 1, 10 };
		double best_lambda = 0, best_bias = 0;
		double best_val = -1;
		for (int i = 0; i < lambda_cand.length; i++) {
			this.lambda = lambda_cand[i];

			this.train(trainSub);

			Result t = this.test(validationSet);
			if (t.macroaverageF > best_val) {
				best_val = t.macroaverageF;
				best_lambda = this.lambda;
				best_bias = this.bias;
			}
			System.out.println("" + "Lambda: " + this.lambda + "\t"
					+ "macroaverage F: " + t.macroaverageF);
		}

		this.lambda = best_lambda;
		this.bias = best_bias;

		System.out.println("\n\n" + "Best_Lambda: " + this.lambda + "\t"
				+ "Best_Bias: " + this.bias + "\t" + "macrosaverafeF: "
				+ best_val);
	}

	double[] initializeParam(File f) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(f));
		String s = br.readLine();
		br.close();
		String[] ss = s.split(" ");
		double[] res = new double[ss.length];
		for (int i = 0; i < ss.length; i++)
			res[i] = Double.valueOf(ss[i]);
		return res;
	}

	void printParam(File f) {
		PrintWriter pr = null;
		try {
			pr = new PrintWriter(new FileWriter(f));
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
		for (int i = 0; i < w.length; i++)
			pr.printf("%f ", w[i]);
		pr.printf("\n");
		pr.close();
	}

	public static void main(String[] args) throws IOException {
		if (args.length < 2) {
			System.err.printf("at least 2 parameters required:\n"
					+ "training data file name\n" + "testing data file name\n");
			return;
		}

		MCML cmlm = new MCML();
		File outputMeasureFile = null;
		boolean selectBestParam = false;

		for (int i = 2; i < args.length; i++) {
			if (args[i].equals("-w")) {
				i++;
				cmlm.paramFile = new File(args[i]);
			} else if (args[i].equals("-sp")) {
				selectBestParam = true;
			} else if (args[i].equals("-ow")) {
				i++;
				cmlm.outputParamFile = new File(args[i]);
			} else if (args[i].equals("-om")) {
				i++;
				outputMeasureFile = new File(args[i]);
			} else if (args[i].equals("-t")) {
				i++;
				cmlm.tolerance = Double.valueOf(args[i]);
			} else if (args[i].equals("-d")) {
				cmlm.debug = true;
			} else {
				System.err.println("unknown flag: " + args[i]);
				return;
			}
		}

		File train_file = new File(args[0]), test_file = new File(args[1]);

		Dataset trainSet = new Dataset(train_file), testSet = new Dataset(
				test_file);
		if (selectBestParam)
			cmlm.selectBestParameter(trainSet);
		cmlm.train(trainSet);
		Result res = cmlm.test(testSet);
		if (outputMeasureFile != null)
			res.printMeasures(outputMeasureFile);
		else
			System.out.println(res);
	}

}
