package sparse;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;

import sparse.Instance.Entry;

import edu.stanford.nlp.optimization.*;

class CRFM {
	double[] w;
	double[] probThreshold;
	// seen label set
	// maps label set to corresponding feature weight index in w[]
	Map<Set<Integer>, Integer> S;

	File paramFile = null;
	File outputParamFile = null;
	double lambda = 1e-4, bias = 1;
	double tolerance = 1e-3;
	double fbr = 0.1;
	boolean debug = false;

	CRFM() {
	}

	CRFM(CRFM crfm) {
		if(crfm.paramFile!=null)
			try {
				this.initializeParam(crfm.paramFile);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		lambda = crfm.lambda;
		bias = crfm.bias;
		if (crfm.probThreshold != null)
			probThreshold = crfm.probThreshold.clone();
		tolerance = crfm.tolerance;
		fbr = crfm.fbr;
	}

	double F(Set<Integer> y, double[] xw_buf) {
		double res = 0;
		for (Integer i : y)
			res += xw_buf[i];
		return res;
	}

	double G(double[] _w, Set<Integer> y) {
		Integer idx = S.get(y);
		return idx != null ? _w[idx] : 0;
	}

	// x is described by xw_buf
	double Z(double[] _w, double[] xw_buf) {
		double res = 1.0;
		for (int i = 0; i < xw_buf.length; i++)
			res *= 1 + Math.exp(xw_buf[i]);
		for (Map.Entry<Set<Integer>, Integer> entry : S.entrySet()) {
			Set<Integer> y = entry.getKey();
			double f = F(y, xw_buf);
			res -= Math.exp(f);
			res += Math.exp(f + _w[entry.getValue()]);
		}
		// System.out.println(res);
		return res;
	}

	double P_Y(double[] _w, Set<Integer> y, double[] xw_buf) {
		return Math.exp(F(y, xw_buf) + G(_w, y)) / Z(_w, xw_buf);
	}

	double[] createXwBuf(double[] _w, Instance ins) {
		double[] xwBuf = new double[ins.L];
		for (int i = 0; i < ins.L; i++)
			xwBuf[i] = ins.dotProdX(_w, i * ins.N);
		return xwBuf;
	}

	double P_Y_l(double[] _w, int l, double[] xw_buf) {
		double down = 1.0, up = 1.0;
		for (int i = 0; i < xw_buf.length; i++)
			down *= 1 + Math.exp(xw_buf[i]);
		up = down * Math.exp(xw_buf[l]) / (1 + Math.exp(xw_buf[l]));
		for (Map.Entry<Set<Integer>, Integer> entry : S.entrySet()) {
			Set<Integer> y = entry.getKey();
			double delta = Math.exp(F(y, xw_buf) + _w[entry.getValue()])
					- Math.exp(F(y, xw_buf));
			if (y.contains(l))
				up += delta;
			down += delta;
		}
		return up / down;
	}

	Queue<Job> toDo = new ConcurrentLinkedQueue<Job>(); // thread-safe
	Queue<Job> done = new ConcurrentLinkedQueue<Job>();

	class Job {
		double[] _w;
		Instance ins;
		double[] res;

		Job(double[] _w, Instance ins) {
			this._w = _w;
			this.ins = ins;
		}

		void doWork() {
			res = new double[_w.length];
			double[] xwBuf = createXwBuf(_w, ins);
			for (int l = 0; l < ins.L; l++) {
				double temp = (ins.y.contains(l) ? 1.0 : 0.0)
						- P_Y_l(_w, l, xwBuf);
				for (Entry e : ins.x)
					res[l * ins.N + e.idx] += temp * e.val;
			}

			for (Map.Entry<Set<Integer>, Integer> entry : S.entrySet()) {
				Set<Integer> y = entry.getKey();
				int j = entry.getValue();
				res[j] += (ins.y.equals(y) ? 1.0 : 0.0) - P_Y(_w, y, xwBuf);
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
			for (int i = 0; i < train.D; i++) {
				Instance ins = train.data.get(i);
				double[] xwBuf = createXwBuf(_w, ins);

				double delta = F(ins.y, xwBuf) + G(_w, ins.y)
						- Math.log(Z(_w, xwBuf));
				res += delta;
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
			toDo.clear();
			done.clear();
			for (int idx = 0; idx < train.D; idx++) {
				Instance ins = train.data.get(idx);
				toDo.add(new Job(_w, ins));
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
				res[i] -= _w[i] * lambda; // regularization
			}
			return res;
		}

	}

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

	void train(Dataset trainSet) throws IOException {
		trainSet.setBias(bias);
		int baseLen = trainSet.L * trainSet.N;
		S = new HashMap<Set<Integer>, Integer>();
		for (int i = 0; i < trainSet.D; i++) {
			Instance ins = trainSet.data.get(i);
			if (!S.containsKey(ins.y)) {
				S.put(ins.y, baseLen + S.size());
			}
		}
		if (paramFile != null)
			w = initializeParam(paramFile);
		else
			w = new double[baseLen + S.size()];
		CGMinimizer optimizer = new CGMinimizer(true);
		DiffFunction f = new NegativeLogLikelihood(new LogLikelihood(trainSet));
		w = optimizer.minimize(f, tolerance, w);

		if (outputParamFile != null)
			this.printParam(outputParamFile);
	}

	// best y given ins.x
	Set<Integer> predict(Instance ins) {
		Set<Integer> res = new HashSet<Integer>();
		double[] xwBuf = this.createXwBuf(w, ins);
		for (int i = 0; i < ins.L; i++)
			if (this.P_Y_l(w, i, xwBuf) > probThreshold[i])
				res.add(i);
		return res;
	}

	Result test(Dataset testSet) {
		if (probThreshold == null || probThreshold.length != testSet.L) {
			probThreshold = new double[testSet.L];
			for (int i = 0; i < testSet.L; i++)
				probThreshold[i] = 0.5;
		}
		testSet.setBias(bias);
		ArrayList<Set<Integer>> pred = new ArrayList<Set<Integer>>();
		for (int i = 0; i < testSet.D; i++)
			pred.add(predict(testSet.data.get(i)));
		return new Result(testSet, pred);
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

	double[] initializeParam(File f) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(f));
		String s = br.readLine();
		br.close();
		String[] ss = s.split(" ");
		double[] theta = new double[ss.length];
		for (int i = 0; i < theta.length; i++)
			theta[i] = Double.valueOf(ss[i]);
		return theta;
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
		int foldNum = 5;
		Dataset[][] CVData = data.createCVData(foldNum);
		CRFM[] crfm = new CRFM[foldNum];
		for (int i = 0; i < foldNum; i++) {
			crfm[i] = new CRFM(this);
			crfm[i].train(CVData[i][0]);
		}
		for (int i = 0; i < L; i++) {
			double sum = 0;
			for (int idx = 0; idx < foldNum; idx++) {
				double[] w = crfm[idx].w;
				ArrayList<Node> arr = new ArrayList<Node>();
				Dataset test = CVData[idx][1];
				for (int j = 0; j < test.D; j++) {
					Instance ins = test.data.get(j);
					double[] xwBuf = this.createXwBuf(w, ins);
					Set<Integer> y = ins.y;
					arr.add(new Node(crfm[idx].P_Y_l(w, i, xwBuf), y
							.contains(i) ? 1 : 0));
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

	void selectBestParameter(Dataset trainSub, Dataset validationSet)
			throws Exception {
		double[] lambda_cand = { 1e-4, 1e-3, 1e-2, 0.1, 1, 10};
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
				+ "macrosaverafeF: " + best_val);

	}

	public void parameterSelection(Dataset trainSet, boolean selectBestParam,
			boolean selectBestThreshold) throws Exception {

		if (selectBestParam) {
			int trainset_size = trainSet.D * 7 / 10;
			Collections.shuffle(trainSet.data);
			Dataset trainSub = new Dataset(trainSet, 0, trainset_size), validationSet = new Dataset(
					trainSet, trainset_size, trainSet.D);
			selectBestParameter(trainSub, validationSet);
		}
		if (selectBestThreshold) {
			selectBestProbThreshold(trainSet);
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
		CRFM crfm = new CRFM();

		for (int i = 2; i < args.length; i++) {
			if (args[i].equals("-w")) {
				i++;
				crfm.paramFile = new File(args[i]);
			} else if (args[i].equals("-sp")) {
				selectBestParam = true;
			} else if (args[i].equals("-st")) {
				selectBestThreshold = true;
			} else if (args[i].equals("-ow")) {
				i++;
				crfm.outputParamFile = new File(args[i]);
			} else if (args[i].equals("-om")) {
				i++;
				outputMeasureFile = new File(args[i]);
			} else if (args[i].equals("-t")) {
				i++;
				crfm.tolerance = Double.valueOf(args[i]);
			} else if (args[i].equals("-d")) {
				crfm.debug = true;
			} else if (args[i].equals("-f")) {
				i++;
				crfm.fbr = Double.valueOf(args[i]);
			} else {
				System.err.println("unknown flag: " + args[i]);
				return;
			}
		}
		File train_file = new File(args[0]), test_file = new File(args[1]);

		Dataset trainSet = new Dataset(train_file), testSet = new Dataset(
				test_file);

		crfm.parameterSelection(trainSet, selectBestParam, selectBestThreshold);
		crfm.train(trainSet);
		Result res = crfm.test(testSet);
		if (outputMeasureFile != null)
			res.printMeasures(outputMeasureFile);
		else
			System.out.println(res);

	}

}
