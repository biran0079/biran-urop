package edu.stanford.nlp.stats;

import edu.stanford.nlp.classify.GeneralDataset;
import edu.stanford.nlp.classify.ProbabilisticClassifier;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.util.BinaryHeapPriorityQueue;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.PriorityQueue;
import edu.stanford.nlp.util.StringUtils;

import java.text.NumberFormat;
import java.util.List;


/**
 * @author Jenny Finkel
 */
public class MultiClassAccuracyStats implements Scorer {
  double[] scores; //sorted scores
  boolean[] isCorrect; // is the i-th example correct
  int numCorrect;
  double logLikelihood;
  double accuracy;
  static String saveFile = null;
  static int saveIndex = 1;

  public static final int USE_ACCURACY = 1;
  public static final int USE_LOGLIKELIHOOD = 2;

  private int scoreType = USE_ACCURACY;


  public MultiClassAccuracyStats(){
  }

  public MultiClassAccuracyStats(int scoreType){
    this.scoreType = scoreType;
  }

  public MultiClassAccuracyStats(String file){
    this(file, USE_ACCURACY);
  }

  public MultiClassAccuracyStats(String file, int scoreType){
    saveFile=file;
    this.scoreType = scoreType;
  }

  public MultiClassAccuracyStats(ProbabilisticClassifier classifier, GeneralDataset data,String file) {
    this(classifier, data, file, USE_ACCURACY);
  }

  public MultiClassAccuracyStats(ProbabilisticClassifier classifier, GeneralDataset data,String file, int scoreType) {
    saveFile=file;
    this.scoreType = scoreType;
    initMC(classifier, data);
  }

  int correct = 0;
  int total = 0;

  public double score(ProbabilisticClassifier classifier, GeneralDataset data) {
      initMC(classifier,data);
      return score();
  }

  public double score() {
    if (scoreType == USE_ACCURACY) {
      return accuracy;
    } else if (scoreType == USE_LOGLIKELIHOOD) {
      return logLikelihood;
    } else {
      throw new RuntimeException("Unknown score type: "+scoreType);
    }
  }

  public int numSamples() {
    return scores.length;
  }

  public double confidenceWeightedAccuracy() {
    double acc = 0;
    for (int recall = 1; recall <= numSamples(); recall++) {
      acc += numCorrect(recall) / (double) recall;
    }
    return acc / numSamples();
  }

  public void initMC(ProbabilisticClassifier classifier, GeneralDataset data) {
    //if (!(gData instanceof Dataset)) {
    //  throw new UnsupportedOperationException("Can only handle Datasets, not "+gData.getClass().getName());
    //}
    //
    //Dataset data = (Dataset)gData;

    PriorityQueue q = new BinaryHeapPriorityQueue();
    total = 0;
    correct = 0;
    logLikelihood = 0.0;
    for (int i = 0; i < data.size(); i++) {
      Datum d = data.getRVFDatum(i);
      ClassicCounter scores = classifier.logProbabilityOf(d);
      Object guess = Counters.argmax(scores);
      Object correctLab = d.label();
      double guessScore = scores.getCount(guess);
      double correctScore = scores.getCount(correctLab);
      int guessInd = data.labelIndex().indexOf(guess);
      int correctInd = data.labelIndex().indexOf(correctLab);

      total++;
      if (guessInd == correctInd) {
        correct++;
      }
      logLikelihood += correctScore;
      q.add(new Pair(new Integer(i), new Pair(new Double(guessScore), new Boolean(guessInd == correctInd))), -guessScore);
    }
    accuracy = (double) correct / (double) total;
    List sorted = q.toSortedList();
    scores = new double[sorted.size()];
    isCorrect = new boolean[sorted.size()];

    for (int i = 0; i < sorted.size(); i++) {
      Pair<Double, Boolean> next = (Pair<Double, Boolean>) (((Pair) sorted.get(i)).second());
      scores[i] = next.first().doubleValue();
      isCorrect[i] = next.second().booleanValue();
    }

  }

  /**
   * how many correct do we have if we return the most confident num recall ones
   *
   * @param recall
   * @return
   */
  public int numCorrect(int recall) {
    int correct = 0;
    for (int j = scores.length - 1; j >= scores.length - recall; j--) {
      if (isCorrect[j]) {
        correct++;
      }
    }
    return correct;
  }


  public int[] getAccCoverage() {
    int[] arr = new int[numSamples()];
    for (int recall = 1; recall <= numSamples(); recall++) {
      arr[recall - 1] = numCorrect(recall);
    }
    return arr;
  }

  public String getDescription(int numDigits) {
    NumberFormat nf = NumberFormat.getNumberInstance();
    nf.setMaximumFractionDigits(numDigits);

    StringBuffer sb = new StringBuffer();
    double confWeightedAccuracy = confidenceWeightedAccuracy();
    sb.append("--- Accuracy Stats ---").append("\n");
    sb.append("accuracy: ").append(nf.format(accuracy)).append(" (").append(correct).append("/").append(total).append(")\n");
    sb.append("confidence weighted accuracy :").append(nf.format(confWeightedAccuracy)).append("\n");
    sb.append("log-likelihood: ").append(logLikelihood).append("\n");
    if (saveFile != null) {
      String f = saveFile + "-" + saveIndex;
      sb.append("saving accuracy info to ").append(f).append(".accuracy\n");
      StringUtils.printToFile(f + ".accuracy", AccuracyStats.toStringArr(getAccCoverage()));
      saveIndex++;
      //sb.append("accuracy coverage: ").append(toStringArr(accrecall)).append("\n");
      //sb.append("optimal accuracy coverage: ").append(toStringArr(optaccrecall));
    }
    return sb.toString();
  }

  public String toString() {
    String accuracyType = null;
    if(scoreType == USE_ACCURACY)
      accuracyType = "classification_accuracy";
    else if(scoreType == USE_LOGLIKELIHOOD)
      accuracyType = "log_likelihood";
    else
      accuracyType = "unknown";
    return "MultiClassAccuracyStats(" + accuracyType  + ")" + scoreType + USE_ACCURACY + USE_LOGLIKELIHOOD;
  }

}
