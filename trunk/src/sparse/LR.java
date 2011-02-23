package sparse;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import sparse.Instance.Entry;

import edu.stanford.nlp.optimization.CGMinimizer;
import edu.stanford.nlp.optimization.DiffFunction;

public class LR {
	File paramFile = null;
	File outputParamFile = null;

	double[][] w = null; // parameters
	double lambda = 0.1, bias = 1.0;
	double[] probThreshold;
	double tolerance = 1e-1;
	double fbr = 0.1;
	boolean debug = false;

	LR() {
	}

	LR(LR lr) {
		if(lr.paramFile!=null)
			try {
				this.initializeParam(lr.paramFile);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		lambda = lr.lambda;
		bias = lr.bias;
		if (lr.probThreshold != null)
			probThreshold = lr.probThreshold.clone();
		tolerance = lr.tolerance;
		fbr = lr.fbr;
	}

	class NegativeLogLikelihood implements DiffFunction {
		LogLikelihood l;

		NegativeLogLikelihood(LogLikelihood l) {
			this.l = l;
		}

		@Override
		public double valueAt(double[] w) {
			return -l.valueAt(w);
		}

		@Override
		public int domainDimension() {
			return l.domainDimension();
		}

		@Override
		public double[] derivativeAt(double[] w) {
			double[] res = l.derivativeAt(w);
			for (int i = 0; i < res.length; i++)
				res[i] = -res[i];
			return res;
		}

	};

	class LogLikelihood implements DiffFunction {

		Dataset train;
		int idx;

		LogLikelihood(Dataset trainSet, int idx) {
			this.train = trainSet;
			this.idx = idx;
		}

		@Override
		public double[] derivativeAt(double[] w) {
			double[] res = new double[w.length];
			for (int i = 0; i < w.length; i++)
				res[i] = 0.0;
			for (int i = 0; i < train.D; i++) {
				Instance ins = train.data.get(i);
				double temp = (ins.y.contains(idx) ? 1.0 : 0.0) - P(w, ins);
				for (Entry e : ins.x)
					res[e.idx] += temp * e.val;
			}
			for (int i = 0; i < res.length; i++) {
				res[i] -= w[i] * lambda;
			}
			return res;
		}

		@Override
		public int domainDimension() {
			return train.N;
		}

		@Override
		public double valueAt(double[] w) {
			double res = 0.0, pxw;
			for (int i = 0; i < train.D; i++) {
				Instance ins = train.data.get(i);
				pxw = P(w, ins);
				res += ins.y.contains(idx) ? Math.log(pxw + 1e-8) : Math
						.log(1 - pxw + 1e-8);
			}
			double penalty = 0.0;
			for (int i = 0; i < w.length; i++)
				penalty = w[i] * w[i];
			penalty *= lambda / 2.0;
			res -= penalty;
			return res;
		}

	}

	class Node implements Comparable<Node> {
		double p;
		int i;

		Node(double p, int i) {
			this.p = p;
			this.i = i;
		}

		@Override
		public int compareTo(Node x) {
			if (p > x.p)
				return 1;
			if (p < x.p)
				return -1;
			return 0;
		}
	}

	void selectBestProbThreshold(Dataset data) throws Exception {
		int L = data.L;
		this.probThreshold = new double[L];
		int foldNum = 10;
		LR[] lr = new LR[foldNum];
		Dataset[][] CVData = data.createCVData(foldNum);
		for (int i = 0; i < foldNum; i++) {
			lr[i] = new LR(this);
			lr[i].train(CVData[i][0]);
			if (debug)
				System.out.println("one LR training finished");
		}
		for (int i = 0; i < L; i++) {
			double sum = 0;
			for (int idx = 0; idx < foldNum; idx++) {
				ArrayList<Node> arr = new ArrayList<Node>();
				Dataset test = CVData[idx][1];
				for (int j = 0; j < test.D; j++) {
					Instance ins = test.data.get(j);
					Set<Integer> y = ins.y;
					arr.add(new Node(lr[idx].P(lr[idx].w[i], ins), y.contains(i) ? 1
							: 0));
				}
				Collections.sort(arr);

				int tp = 0, tn = 0, fp = 0, fn = 0;
				for (int j = 0; j < arr.size(); j++) {
					if (arr.get(j).i == 0)
						fp++;
					else
						tp++;
				}
				double bestF = 0, bestP = arr.get(0).p - 1e-6;
				for (int j = 0; j < arr.size(); j++) {
					if (arr.get(j).i == 1) {
						tp--;
						fn++;
					} else {
						tn++;
						fp--;
					}
					double f = (2.0 * tp + fp + fn) == 0 ? 0 : 2.0 * tp
							/ (2.0 * tp + fp + fn);

					if (f > bestF) {
						bestF = f;
						if (j == arr.size() - 1)
							bestP = arr.get(j).p + 1e-6;
						else
							bestP = (arr.get(j).p + arr.get(j + 1).p) / 2.0;
					}
				}
				if (bestF > fbr)
					sum += bestP;
				else
					sum += arr.get(arr.size() - 1).p;
			}

			this.probThreshold[i] = sum / foldNum;
		}
	}

	Result crossValidation(Dataset data, int foldNum) throws Exception {
		Result[] res = new Result[foldNum];
		Dataset[][] CVData = data.createCVData(foldNum);
		for (int i = 0; i < foldNum; i++) {
			train(CVData[i][0]);
			if (debug)
				System.out.println("one LR training finished");
			res[i] = test(CVData[i][1]);
		}
		return Result.average(res);
	}

	void selectBestParameter(Dataset data) throws Exception {

		double[] lambda_cand = { 1e-3, 1e-2, 1e-1, 1, 10 };
		double best_lambda = 0, best_bias = 0;
		double best_val = -1;
		for (int i = 0; i < lambda_cand.length; i++) {
			this.lambda = lambda_cand[i];

			Result t = this.crossValidation(data, 5);
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
				+ "macrosaverafeF: " + best_val);
	}

	public double P(double[] w, Instance ins) {
		double xw = Math.exp(ins.dotProdX(w, 0));
		return xw / (1.0 + xw);
	}

	double[][] initializeParam(File f) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(f));
		String s = br.readLine();
		String[] ss = s.split(" ");
		int L = Integer.valueOf(ss[0]), N = Integer.valueOf(ss[1]);
		double[][] theta = new double[L][N];
		s = br.readLine();
		ss = s.split(" ");
		br.close();
		for (int i = 0; i < L; i++)
			for (int j = 0; j < N; j++)
				theta[i][j] = Double.valueOf(ss[i * N + j]);
		return theta;
	}

	void train(Dataset trainSet) throws Exception {
		trainSet.setBias(bias);

		int L = trainSet.L;
		int N = trainSet.N;

		if (paramFile != null)
			w = initializeParam(paramFile);
		else if (w == null || w.length != L || w[0].length != N)
			w = new double[L][N];

		CGMinimizer opt = new CGMinimizer(true);
		for (int l = 0; l < trainSet.L; l++) {
			DiffFunction f = new NegativeLogLikelihood(new LogLikelihood(
					trainSet, l));
			w[l] = opt.minimize(f, tolerance, w[l]);
		}
	}

	void printParam(File f) {
		PrintWriter pr = null;
		try {
			pr = new PrintWriter(new FileWriter(f));
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
		pr.printf("%d %d\n", w.length, w[0].length);
		for (int i = 0; i < w.length; i++)
			for (int j = 0; j < w[i].length; j++)
				pr.printf("%f ", w[i][j]);
		pr.printf("\n");
		pr.close();
	}

	Result test(Dataset testSet) {
		testSet.setBias(bias);
		if (probThreshold == null || probThreshold.length != testSet.L) {
			probThreshold = new double[testSet.L];
			for (int i = 0; i < testSet.L; i++)
				probThreshold[i] = 0.5;
		}
		ArrayList<Set<Integer>> pred = new ArrayList<Set<Integer>>();
		for (int i = 0; i < testSet.D; i++)
			pred.add(predict(w, testSet.data.get(i)));
		return new Result(testSet, pred);
	}

	Set<Integer> predict(double[][] w, Instance ins) {
		Set<Integer> res = new HashSet<Integer>();
		for (int l = 0; l < ins.L; l++) {
			if (P(w[l], ins) > probThreshold[l])
				res.add(l);
		}
		return res;
	}

	void initializeParam(File f, double[][] X, double[][] Y) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(f));
		String s = br.readLine();
		br.close();
		String[] ss = s.split(" ");
		w = new double[Y[0].length][X[0].length];
		for (int i = 0; i < w.length; i++)
			for (int j = 0; j < w[i].length; j++)
				w[i][j] = Double.valueOf(ss[i * w[i].length + j]);
	}

	public void parameterSelection(Dataset trainSet, boolean selectBestParam,
			boolean selectBestThreshold) throws Exception {

		if (selectBestThreshold) {
			selectBestProbThreshold(trainSet);
		}
		if (selectBestParam) {
			selectBestParameter(trainSet);
		}
	}

	public static void main(String[] args) throws Exception {
		if (args.length < 2) {
			System.err.printf("at least 2 parameters required:\n"
					+ "training data file name\n" + "testing data file name\n");
			return;
		}

		File outputMeasureFile = null;
		boolean selectBestParam = false;
		boolean selectBestThreshold = false;

		LR lr = new LR();

		for (int i = 2; i < args.length; i++) {
			if (args[i].equals("-w")) {
				i++;
				lr.paramFile = new File(args[i]);
			} else if (args[i].equals("-sp")) {
				selectBestParam = true;
			} else if (args[i].equals("-st")) {
				selectBestThreshold = true;
			} else if (args[i].equals("-ow")) {
				i++;
				lr.outputParamFile = new File(args[i]);
			} else if (args[i].equals("-om")) {
				i++;
				outputMeasureFile = new File(args[i]);
			} else if (args[i].equals("-t")) {
				i++;
				lr.tolerance = Double.valueOf(args[i]);
			} else if (args[i].equals("-d")) {
				lr.debug = true;
			} else if (args[i].equals("-f")) {
				i++;
				lr.fbr = Double.valueOf(args[i]);
			} else {
				System.err.println("unknown flag: " + args[i]);
				return;
			}
		}
		File train_file = new File(args[0]), test_file = new File(args[1]);

		Dataset trainSet = new Dataset(train_file), testSet = new Dataset(
				test_file);

		lr.parameterSelection(trainSet, selectBestParam, selectBestThreshold);
		lr.train(trainSet);
		Result res = lr.test(testSet);
		if (outputMeasureFile != null)
			res.printMeasures(outputMeasureFile);
		System.out.println(res);
	}

}
