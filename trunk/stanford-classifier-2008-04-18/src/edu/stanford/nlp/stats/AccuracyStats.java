package edu.stanford.nlp.stats;

import edu.stanford.nlp.classify.GeneralDataset;
import edu.stanford.nlp.classify.PRCurve;
import edu.stanford.nlp.classify.ProbabilisticClassifier;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;

import java.text.NumberFormat;
import java.util.ArrayList;

/**
 * Utility class for aggregating counts of true positives, false positives, and
 * false negatives and computing precision/recall/F1 stats. Can be used for a single
 * collection of stats, or to aggregate stats from a bunch of runs.
 *
 * @author Kristina Toutanova
 * @author Jenny Finkel
 */
public class AccuracyStats implements Scorer {

  double confWeightedAccuracy;
  double accuracy;
  double optAccuracy;
  double optConfWeightedAccuracy;
  double logLikelihood;
  int[] accrecall;
  int[] optaccrecall;

  Object posLabel;

  String saveFile = null;
  static int saveIndex = 1;

  public AccuracyStats(ProbabilisticClassifier classifier, GeneralDataset data, Object posLabel) {
    this.posLabel = posLabel;
    score(classifier, data);
  }

  public AccuracyStats(Object posLabel, String saveFile) {
    this.posLabel = posLabel;
    this.saveFile = saveFile;
  }

  public double score(ProbabilisticClassifier classifier, GeneralDataset data) {

    Index labelIndex = data.labelIndex;


    ArrayList dataScores = new ArrayList<Pair<Double, Integer>>();
    for (int i = 0; i < data.size(); i++) {
      Datum d = data.getRVFDatum(i);
      ClassicCounter scores = classifier.logProbabilityOf(d);
      int labelD = d.label().equals(posLabel) ? 1 : 0;
      dataScores.add(new Pair<Double, Integer>(Math.exp(scores.getCount(posLabel)), labelD));
    }

    PRCurve prc = new PRCurve(dataScores);

    confWeightedAccuracy = prc.cwa();
    accuracy = prc.accuracy();
    optAccuracy = prc.optimalAccuracy();
    optConfWeightedAccuracy = prc.optimalCwa();
    logLikelihood = prc.logLikelihood();
    accrecall = prc.cwaArray();
    optaccrecall = prc.optimalCwaArray();

    return accuracy;
  }

  public String getDescription(int numDigits) {
    NumberFormat nf = NumberFormat.getNumberInstance();
    nf.setMaximumFractionDigits(numDigits);

    StringBuffer sb = new StringBuffer();
    sb.append("--- Accuracy Stats ---").append("\n");
    sb.append("accuracy: ").append(nf.format(accuracy)).append("\n");
    sb.append("optimal fn accuracy: ").append(nf.format(optAccuracy)).append("\n");
    sb.append("confidence weighted accuracy :").append(nf.format(confWeightedAccuracy)).append("\n");
    sb.append("optimal confidence weighted accuracy: ").append(nf.format(optConfWeightedAccuracy)).append("\n");
    sb.append("log-likelihood: ").append(logLikelihood).append("\n");
    if (saveFile != null) {
      String f = saveFile + "-" + saveIndex;
      sb.append("saving accuracy info to ").append(f).append(".accuracy\n");
      StringUtils.printToFile(f + ".accuracy", toStringArr(accrecall));
      sb.append("saving optimal accuracy info to ").append(f).append(".optimal_accuracy\n");
      StringUtils.printToFile(f + ".optimal_accuracy", toStringArr(optaccrecall));
      saveIndex++;
      //sb.append("accuracy coverage: ").append(toStringArr(accrecall)).append("\n");
      //sb.append("optimal accuracy coverage: ").append(toStringArr(optaccrecall));
    }
    return sb.toString();
  }

  public static String toStringArr(int[] acc) {
    StringBuffer sb = new StringBuffer();
    int total = acc.length;
    NumberFormat nf = NumberFormat.getInstance();
    for (int i = 0; i < acc.length; i++) {
      double coverage = (i + 1) / (double) total;
      double accuracy = acc[i] / (double) (i + 1);
      coverage *= 1000000;
      accuracy *= 1000000;
      coverage = (int) coverage;
      accuracy = (int) accuracy;
      coverage /= 10000;
      accuracy /= 10000;
      sb.append(coverage);
      sb.append("\t");
      sb.append(accuracy);
      sb.append("\n");
    }
    return sb.toString();
  }

}
