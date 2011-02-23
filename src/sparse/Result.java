package sparse;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Set;

public class Result {
	int dataset_size, label_number;

	private Result() {
	}

	Result(Dataset test, ArrayList<Set<Integer>> pred) {
		dataset_size = test.D;
		label_number = test.L;
		measure(test, pred);
	}

	static Result average(Result[] l) {
		Result res = new Result();
		for (int i = 0; i < l.length; i++) {
			res.accuracy += l[i].accuracy;
			res.hamming += l[i].hamming;
			res.exact_match_ratio += l[i].exact_match_ratio;
			res.precision += l[i].precision;
			res.recall += l[i].recall;
			res.macroaverageF += l[i].macroaverageF;
			res.microaverageF += l[i].microaverageF;
		}
		for (int i = 0; i < l.length; i++) {
			res.accuracy /= l.length;
			res.hamming /= l.length;
			res.exact_match_ratio /= l.length;
			res.precision /= l.length;
			res.recall /= l.length;
			res.macroaverageF /= l.length;
			res.microaverageF /= l.length;
		}
		return res;
	}

	double accuracy, hamming, exact_match_ratio, precision, recall,
			macroaverageF, microaverageF;

	public int countIntersection(Set<Integer> s1, Set<Integer> s2) {
		int res = 0;
		for (Integer i : s1)
			if (s2.contains(i))
				res++;
		return res;
	}

	public int countUnion(Set<Integer> s1, Set<Integer> s2) {
		return s1.size() + s2.size() - countIntersection(s1, s2);
	}

	public int countXor(Set<Integer> s1, Set<Integer> s2) {
		return countUnion(s1, s2) - countIntersection(s1, s2);
	}

	private void measure(Dataset test, ArrayList<Set<Integer>> pred) {
		accuracy = 0;
		hamming = 0;
		exact_match_ratio = 0;

		for (int i = 0; i < test.D; i++) {
			Set<Integer> prediction = pred.get(i);
			Set<Integer> correct = test.data.get(i).y;
			if (prediction.equals(correct)) {
				exact_match_ratio += 1;
			}
			int and_len = countIntersection(prediction, correct);
			int or_len = countUnion(prediction, correct);
			int xor_len = countXor(prediction, correct);

			if (or_len == 0) {
				accuracy += 1;
			} else {
				accuracy += (double) and_len / or_len;
			}
			hamming += xor_len;
		}
		hamming /= label_number * dataset_size;
		accuracy /= dataset_size;
		exact_match_ratio /= dataset_size;
		int tp_sum = 0, fp_sum = 0, fn_sum = 0;
		double F = 0;
		for (int j = 0; j < label_number; j++) {
			int tp = 0, fp = 0, fn = 0, tn = 0;
			for (int i = 0; i < dataset_size; i++) {
				Set<Integer> prediction = pred.get(i);
				Set<Integer> correct = test.data.get(i).y;
				if (correct.contains(j) && prediction.contains(j)) {
					tp++;
				} else if (!correct.contains(j) && prediction.contains(j)) {
					fp++;
				} else if (correct.contains(j) && !prediction.contains(j)) {
					fn++;
				} else {
					tn++;
				}
			}
			if (tp != 0 || fp != 0 || fn != 0) {
				F += 2.0 * tp / (2.0 * tp + fp + fn);
			}
			tp_sum += tp;
			fp_sum += fp;
			fn_sum += fn;
		}
		this.precision = (double) tp_sum / (tp_sum + fp_sum);
		this.recall = (double) tp_sum / (tp_sum + fn_sum);
		this.microaverageF = 2 * this.precision * this.recall
				/ (this.precision + this.recall);
		this.macroaverageF = F / label_number;
	}

	public void printMeasures(File f) {
		PrintWriter pr = null;
		try {
			pr = new PrintWriter(new FileWriter(f));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		String res = "Exact match ratio: " + this.exact_match_ratio + "\n"
				+ "Accuracy: " + this.accuracy + "\n" + "Precision: "
				+ this.precision + "\n" + "Recall: " + this.recall + "\n"
				+ "Hamming los: " + this.hamming + "\n"
				+ "Microaverage F-measure: " + this.microaverageF + "\n"
				+ "Macroaverage F-measure: " + this.macroaverageF + "\n";
		pr.print(res);
		pr.close();
	}

	@Override
	public String toString() {
		return "Exact match ratio: " + this.exact_match_ratio + "\n"
				+ "Accuracy: " + this.accuracy + "\n" + "Precision: "
				+ this.precision + "\n" + "Recall: " + this.recall + "\n"
				+ "Hamming los: " + this.hamming + "\n"
				+ "Microaverage F-measure: " + this.microaverageF + "\n"
				+ "Macroaverage F-measure: " + this.macroaverageF + "\n";

	}
}
