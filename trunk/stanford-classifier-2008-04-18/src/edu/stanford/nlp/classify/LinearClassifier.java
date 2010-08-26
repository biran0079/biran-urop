// Stanford Classifier - a multiclass maxent classifier
// LinearClassifier
// Copyright (c) 2003-2007 The Board of Trustees of
// The Leland Stanford Junior University. All Rights Reserved.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//
// For more information, bug reports, fixes, contact:
//    Christopher Manning
//    Dept of Computer Science, Gates 1A
//    Stanford CA 94305-9010
//    USA
//    java-nlp-support@lists.stanford.edu
//    http://www-nlp.stanford.edu/software/classifier.shtml

package edu.stanford.nlp.classify;

import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.math.ArrayMath;
import edu.stanford.nlp.util.Function;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Distribution;
import edu.stanford.nlp.stats.Counters;

import java.io.*;
import java.util.zip.GZIPInputStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;


/**
 * Implements a multiclass linear classifier. At classification time this
 * can be any generalized linear model classifier (such as a perceptron,
 * naive logistic regression, SVM).
 *
 * @author Dan Klein
 * @author Jenny Finkel
 * @author Galen Andrew (converted to arrays and indices)
 * @author Christopher Manning (most of the printing options)
 * @author Eric Yeh (save to text file, new constructor w/thresholds)
 */
public class LinearClassifier implements ProbabilisticClassifier, RVFClassifier {

  /** Classifier weights. First index is the featureIndex value and second
   *  index is the labelIndex value.
   */
  private double[][] weights;
  private Index<Object> labelIndex;
  private Index<Object> featureIndex;
  public boolean intern = false;   // variable should be deleted when breaking serialization anyway....
  private double[] thresholds = null;

  private static final long serialVersionUID = 8499574525453275255L;

  private static final int MAX_FEATURE_ALIGN_WIDTH = 50;

  public static final String TEXT_SERIALIZATION_DELIMITER = "\t";
  
  public Collection<Object> labels() {
    return labelIndex.objectsList();
  }

  public Collection<Object> features() {
    return featureIndex.objectsList();
  }

  public Index<Object> labelIndex() {
    return labelIndex;
  }

  public Index featureIndex() {
    return featureIndex;
  }

  private double weight(int iFeature, int iLabel) {
    if (iFeature < 0) {
      //System.err.println("feature not seen ");
      return 0.0;
    } else {
      return weights[iFeature][iLabel];
    }
  }

  private double weight(Object feature, int iLabel) {
    int f = featureIndex.indexOf(feature);
    return weight(f, iLabel);
  }

  public double weight(Object feature, Object label) {
    int f = featureIndex.indexOf(feature);
    int iLabel = labelIndex.indexOf(label);
    return weight(f, iLabel);
  }

  /* --- obsolete method from before this class was rewritten using arrays
  public Counter scoresOf(Datum example) {
    Counter scores = new Counter();
    for (Object l : labels()) {
      scores.setCount(l, scoreOf(example, l));
    }
    return scores;
  }
  --- */

  /** Construct a counter with keys the labels of the classifier and
   *  values the score (unnormalized log probability) of each class.
   */
  public ClassicCounter scoresOf(Datum example) {
    Collection feats = example.asFeatures();
    int[] features = new int[feats.size()];
    int i = 0;
    for (Object f : feats) {
      int index = featureIndex.indexOf(f);
      if (index >= 0) {
        features[i++] = index;
      } else {
        //System.err.println("FEATURE LESS THAN ZERO: " + f);
      }
    }
    int[] activeFeatures = new int[i];
    System.arraycopy(features, 0, activeFeatures, 0, i);
    ClassicCounter<Object> scores = new ClassicCounter<Object>();
    for (Object lab : labels()) {
      scores.setCount(lab, scoreOf(activeFeatures, lab));
    }
    return scores;
  }

  /** Returns of the score of the Datum for the specified label.  
   *  Ignores the true label of the Datum.
   */
  public double scoreOf(Datum example, Object label) {
    int iLabel = labelIndex.indexOf(label);
    double score = 0.0;
    for (Object f : example.asFeatures()) {
      score += weight(f, iLabel);
    }
    return score + thresholds[iLabel];
  }

  /** Construct a counter with keys the labels of the classifier and
   *  values the score (unnormalized log probability) of each class 
   *  for an RVFDatum.
   */
  public ClassicCounter scoresOf(RVFDatum example) {
    ClassicCounter<Object> scores = new ClassicCounter<Object>();
    for (Object l : labels()) {
      scores.setCount(l, scoreOf(example, l));
    }
    //System.out.println("Scores are: " + scores + "   (gold: " + example.label() + ")");
    return scores;
  }

  /** Returns the score of the RVFDatum for the specified label.
   *  Ignores the true label of the RVFDatum.
   */
  public double scoreOf(RVFDatum example, Object label) {
    int iLabel = labelIndex.indexOf(label);
    double score = 0.0;
    ClassicCounter<Object> features = example.asFeaturesCounter();
    for (Object f : features.keySet()) {
      score += weight(f, iLabel) * features.getCount(f);
    }
    return score + thresholds[iLabel];
  }

  /** Returns of the score of the Datum as internalized features for the
   *  specified label. Ignores the true label of the Datum.
   *  Doesn't consider a value for each feature.
   */
  private double scoreOf(int[] feats, Object label) {
    int iLabel = labelIndex.indexOf(label);
    double score = 0.0;
    for (int feat : feats) {
      score += weight(feat, iLabel);
    }
    return score + thresholds[iLabel];
  }


  /**
   * Returns a counter mapping from each class name to the probability of 
   * that class for a certain example.
   * Looking at the the sum of each count v, should be 1.0.
   */
  public ClassicCounter probabilityOf(Datum example) {
    ClassicCounter scores = logProbabilityOf(example);
    for (Object label : scores.keySet()) {
      scores.setCount(label, Math.exp(scores.getCount(label)));
    }
    return scores;
  }

  /**
   * Returns a counter mapping from each class name to the probability of
   * that class for a certain example.
   * Looking at the the sum of each count v, should be 1.0.
   */
  public ClassicCounter probabilityOf(RVFDatum example) {
    // NB: this duplicate method is needed so it calls the scoresOf method
    // with a RVFDatum signature
    ClassicCounter scores = logProbabilityOf(example);  
    for (Object label : scores.keySet()) {
      scores.setCount(label, Math.exp(scores.getCount(label)));
    }
    return scores;
  }

  /**
   * Returns a counter mapping from each class name to the log probability of
   * that class for a certain example.
   * Looking at the the sum of e^v for each count v, should be 1.0.
   */
  public ClassicCounter logProbabilityOf(Datum example) {
    ClassicCounter scores = scoresOf(example);
    double[] scoreArray = new double[scores.keySet().size()];
    int i = 0;
    for (Object k : scores.keySet()) {
      scoreArray[i++] = scores.getCount(k);
    }
    double sum = ArrayMath.logSum(scoreArray);
    scores.incrementAll(-sum);
    return scores;
  }

  /**
   * Returns a counter for the log probability of each of the classes
   * looking at the the sum of e^v for each count v, should be 1
   */
  public ClassicCounter logProbabilityOf(RVFDatum example) {
    // NB: this duplicate method is needed so it calls the scoresOf method
    // with an RVFDatum signature!!
    ClassicCounter scores = scoresOf(example);
    double[] scoreArray = new double[scores.keySet().size()];
    int i = 0;
    for (Object k : scores.keySet()) {
      scoreArray[i++] = scores.getCount(k);
    }
    double sum = ArrayMath.logSum(scoreArray);
    scores.incrementAll(-sum);
    return scores;
  }

  /** Return a String that prints features with large weights.
   *
   * @param useMagnitude Whether the notion of "large" should ignore
   *                     the sign of the feature weight.
   * @param numFeatures  How many top features to print
   * @return The String representation of features with large weights
   */
  public String toBiggestWeightFeaturesString(boolean useMagnitude,
      int numFeatures,
      boolean printDescending) {
    // this used to try to use a treeset, but that was WRONG....
    edu.stanford.nlp.util.PriorityQueue<Pair<Integer,Integer>> biggestKeys = 
      new FixedPrioritiesPriorityQueue<Pair<Integer,Integer>>();

    // locate biggest keys
    for (int feat = 0; feat < weights.length; feat++) {
      for (int lab = 0; lab < weights[feat].length; lab++) {
        double thisWeight;
        // reverse the weight, so get smallest first
        if (useMagnitude) {
          thisWeight = -Math.abs(weights[feat][lab]);
        } else {
          thisWeight = -weights[feat][lab];
        }
        if (biggestKeys.size() == numFeatures) {
          // have enough features, add only if bigger
          double lowest = biggestKeys.getPriority();
          if (thisWeight < lowest) {
            // remove smallest
            biggestKeys.removeFirst();
            biggestKeys.add(new Pair<Integer, Integer>(feat, lab), thisWeight);
          }
        } else {
          // always add it if don't have enough features yet
          biggestKeys.add(new Pair<Integer, Integer>(feat, lab), thisWeight);
        }
      }
    }

    // Put in List either reversed or not
    // (Note: can't repeatedly iterate over PriorityQueue.)
    int actualSize = biggestKeys.size();
    Pair<Integer, Integer>[] bigArray = new Pair[actualSize];
    // System.err.println("biggestKeys is " + biggestKeys);
    if (printDescending) {
      for (int j = actualSize - 1; j >= 0; j--) {
        bigArray[j] = biggestKeys.removeFirst();
      }
    } else {
      for (int j = 0; j < actualSize; j--) {
        bigArray[j] = biggestKeys.removeFirst();
      }
    }
    List<Pair<Integer, Integer>> bigColl = Arrays.asList(bigArray);
    // System.err.println("bigColl is " + bigColl);

    // find longest key length (for pretty printing) with a limit
    int maxLeng = 0;
    for (Pair<Integer,Integer> p : bigColl) {
      String key = "(" + featureIndex.get(p.first) + "," + labelIndex.get(p.second) + ")";
      int leng = key.length();
      if (leng > maxLeng) {
        maxLeng = leng;
      }
    }
    maxLeng = Math.min(64, maxLeng);

    // set up pretty printing of weights
    NumberFormat nf = NumberFormat.getNumberInstance();
    nf.setMinimumFractionDigits(4);
    nf.setMaximumFractionDigits(4);
    if (nf instanceof DecimalFormat) {
      ((DecimalFormat) nf).setPositivePrefix(" ");
    }

    //print high weight features to a String
    StringBuilder sb = new StringBuilder("LinearClassifier [printing top " + numFeatures + " features]\n");
    for (Pair<Integer, Integer> p : bigColl) {
      String key = "(" + featureIndex.get(p.first) + "," + labelIndex.get(p.second) + ")";
      sb.append(StringUtils.pad(key, maxLeng));
      sb.append(" ");
      double cnt = weights[p.first][p.second];
      if (Double.isInfinite(cnt)) {
        sb.append(cnt);
      } else {
        sb.append(nf.format(cnt));
      }
      sb.append("\n");
    }
    return sb.toString();
  }

  /**
   * Similar to histogram but exact values of the weights
   * to see whether there are many equal weights.
   *
   * @return A human readable string about the classifier distribution.
   */
  public String toDistributionString(int treshold) {
    ClassicCounter<Double> weightCounts = new ClassicCounter<Double>();
    StringBuilder s = new StringBuilder();
    s.append("Total number of weights: ").append(totalSize());
    for (int f = 0; f < weights.length; f++) {
      for (int l = 0; l < weights[f].length; l++) {
        weightCounts.incrementCount(weights[f][l]);
      }
    }

    s.append("Counts of weights\n");
    Set keys = Counters.keysAbove(weightCounts, (double) treshold);
    s.append(keys.size() + " keys occur more than " + treshold + " times ");
    return s.toString();
  }

  public int totalSize() {
    return labelIndex.size() * featureIndex.size();
  }

  public String toHistogramString() {
    // big classifiers
    double[][] hist = new double[3][202];
    Object[][] histEg = new Object[3][202];
    int num = 0;
    int pos = 0;
    int neg = 0;
    int zero = 0;
    double total = 0.0;
    double x2total = 0.0;
    double max = 0.0, min = 0.0;
    for (int f = 0; f < weights.length; f++) {
      for (int l = 0; l < weights[f].length; l++) {
        Object feat = new Pair(featureIndex.get(f), labelIndex.get(l));
        num++;
        double wt = weights[f][l];
        total += wt;
        x2total += wt * wt;
        if (wt > max) {
          max = wt;
        }
        if (wt < min) {
          min = wt;
        }
        if (wt < 0.0) {
          neg++;
        } else if (wt > 0.0) {
          pos++;
        } else {
          zero++;
        }
        int index;
        index = bucketizeValue(wt);
        hist[0][index]++;
        if (histEg[0][index] == null) {
          histEg[0][index] = feat;
        }
        if (wt < 0.1 && wt >= -0.1) {
          index = bucketizeValue(wt * 100.0);
          hist[1][index]++;
          if (histEg[1][index] == null) {
            histEg[1][index] = feat;
          }
          if (wt < 0.001 && wt >= -0.001) {
            index = bucketizeValue(wt * 10000.0);
            hist[2][index]++;
            if (histEg[2][index] == null) {
              histEg[2][index] = feat;
            }
          }
        }
      }
    }
    double ave = total / num;
    double stddev = (x2total / num) - ave * ave;
    StringWriter sw = new StringWriter();
    PrintWriter pw = new PrintWriter(sw);

    pw.println("Linear classifier with " + num + " f(x,y) features");
    pw.println("Average weight: " + ave + "; std dev: " + stddev);
    pw.println("Max weight: " + max + " min weight: " + min);
    pw.println("Weights: " + neg + " negative; " + pos + " positive; " + zero + " zero.");

    printHistCounts(0, "Counts of lambda parameters between [-10, 10)", pw, hist, histEg);
    printHistCounts(1, "Closeup view of [-0.1, 0.1) depicted * 10^2", pw, hist, histEg);
    printHistCounts(2, "Closeup view of [-0.001, 0.001) depicted * 10^4", pw, hist, histEg);
    pw.close();
    return sw.toString();
  }

  /** Print out a partial representation of a linear classifier.
   *  This just calls toString("WeightHistogram", 0)
   */
  public String toString() {
    return toString("WeightHistogram", 0);
  }


  /**
   * Print out a partial representation of a linear classifier in one of
   * several ways.
   *
   * @param style Options are:
   *              HighWeight: print out the param parameters with largest weights;
   *              HighMagnitude: print out the param parameters for which the absolute
   *              value of their weight is largest;
   *              AllWeights: print out the weights of all features
   *              WeightHistogram: print out a particular hard-coded textual histogram
   *              representation of a classifier
   * @param param Determines the number of things printed in certain styles
   * @throws IllegalArgumentException if the style name is unrecognized
   */
  public String toString(String style, int param) {
    if (style.equalsIgnoreCase("HighWeight")) {
      return toBiggestWeightFeaturesString(false, param, true);
    } else if (style.equalsIgnoreCase("HighMagnitude")) {
      return toBiggestWeightFeaturesString(true, param, true);
    } else if (style.equalsIgnoreCase("AllWeights")) {
      return toAllWeightsString();
    } else if (style.equalsIgnoreCase("WeightHistogram")) {
      return toHistogramString();
    } else if (style.equalsIgnoreCase("WeightDistribution")) {
      return toDistributionString(param);
    } else {
      throw new IllegalArgumentException("Unknown style: " + style);
    }
  }


  /**
   * Convert parameter value into number between 0 and 201
   */
  private int bucketizeValue(double wt) {
    int index;
    if (wt >= 0.0) {
      index = ((int) (wt * 10.0)) + 100;
    } else {
      index = ((int) (Math.floor(wt * 10.0))) + 100;
    }
    if (index < 0) {
      index = 201;
    } else if (index > 200) {
      index = 200;
    }
    return index;
  }

  /**
   * Print histogram counts from hist and examples over a certain range
   */
  private void printHistCounts(int ind, String title, PrintWriter pw, double[][] hist, Object[][] histEg) {
    pw.println(title);
    for (int i = 0; i < 200; i++) {
      int intpart, fracpart;
      if (i < 100) {
        intpart = 10 - ((i + 9) / 10);
        fracpart = (10 - (i % 10)) % 10;
      } else {
        intpart = (i / 10) - 10;
        fracpart = i % 10;
      }
      pw.print("[" + ((i < 100) ? "-" : "") + intpart + "." + fracpart + ", " + ((i < 100) ? "-" : "") + intpart + "." + fracpart + "+0.1): " + hist[ind][i]);
      if (histEg[ind][i] != null) {
        pw.print("  [" + histEg[ind][i] + ((hist[ind][i] > 1) ? ", ..." : "") + "]");
      }
      pw.println();
    }
  }


  public String toAllWeightsString() {
    StringWriter sw = new StringWriter();
    PrintWriter pw = new PrintWriter(sw);
    pw.println("Linear classifier with the following weights");
    Datum allFeatures = new BasicDatum(features(), "");
    justificationOf(allFeatures, pw);
    return sw.toString();
  }


  /**
   * Print all features in the classifier and the weight that they assign
   * to each class.
   */
  public void dump() {
    Datum allFeatures = new BasicDatum(features(), "");
    justificationOf(allFeatures);
  }

  public void dump(PrintWriter pw) {
    Datum allFeatures = new BasicDatum(features(), "");
    justificationOf(allFeatures, pw);
  }

  public void justificationOf(RVFDatum example) {
    PrintWriter pw = new PrintWriter(System.err, true);
    justificationOf(example, pw);
  }


  /**
   * Print all features active for a particular datum and the weight that
   * the classifier assigns to each class for those features.
   */
  public void justificationOf(RVFDatum example, PrintWriter pw) {
    int featureLength = 0;
    int labelLength = 6;
    NumberFormat nf = NumberFormat.getNumberInstance();
    nf.setMinimumFractionDigits(2);
    nf.setMaximumFractionDigits(2);
    if (nf instanceof DecimalFormat) {
      ((DecimalFormat) nf).setPositivePrefix(" ");
    }
    ClassicCounter features = example.asFeaturesCounter();
    for (Object f : features.keySet()) {
      featureLength = Math.max(featureLength, f.toString().length() + 2 +
          nf.format(features.getCount(f)).length());
    }
    // make as wide as total printout
    featureLength = Math.max(featureLength, "Total:".length());
    // don't make it ridiculously wide
    featureLength = Math.min(featureLength, MAX_FEATURE_ALIGN_WIDTH);

    for (Object l : labels()) {
      labelLength = Math.max(labelLength, l.toString().length());
    }

    StringBuilder header = new StringBuilder("");
    for (int s = 0; s < featureLength; s++) {
      header.append(' ');
    }
    for (Object l : labels()) {
      header.append(' ');
      header.append(StringUtils.pad(l, labelLength));
    }
    pw.println(header);
    for (Object f : features.keySet()) {
      String fStr = f.toString();
      StringBuilder line = new StringBuilder(fStr);
      line.append("[").append(nf.format(features.getCount(f))).append("]");
      fStr = line.toString();
      for (int s = fStr.length(); s < featureLength; s++) {
        line.append(' ');
      }
      for (Object l : labels()) {
        String lStr = nf.format(weight(f, l));
        line.append(' ');
        line.append(lStr);
        for (int s = lStr.length(); s < labelLength; s++) {
          line.append(' ');
        }
      }
      pw.println(line);
    }
    ClassicCounter scores = scoresOf(example);
    StringBuilder footer = new StringBuilder("Total:");
    for (int s = footer.length(); s < featureLength; s++) {
      footer.append(' ');
    }
    for (Object l : labels()) {
      footer.append(' ');
      String str = nf.format(scores.getCount(l));
      footer.append(str);
      for (int s = str.length(); s < labelLength; s++) {
        footer.append(' ');
      }
    }
    pw.println(footer);
    Distribution distr = Distribution.distributionFromLogisticCounter(scores);
    footer = new StringBuilder("Prob:");
    for (int s = footer.length(); s < featureLength; s++) {
      footer.append(' ');
    }
    for (Object l : labels()) {
      footer.append(' ');
      String str = nf.format(distr.getCount(l));
      footer.append(str);
      for (int s = str.length(); s < labelLength; s++) {
        footer.append(' ');
      }
    }
    pw.println(footer);
  }


  public void justificationOf(Datum example) {
    PrintWriter pw = new PrintWriter(System.err, true);
    justificationOf(example, pw);
  }

  public void justificationOf(Datum example, PrintWriter pw, Function printer) {
    justificationOf(example, pw, printer, false);
  }

  /** Print all features active for a particular datum and the weight that
   *  the classifier assigns to each class for those features.
   *
   *  @param example The datum for which features are to be printed
   *  @param pw Where to print it to
   *  @param printer If this is non-null, then it is applied to each
   *        feature to convert it to a more readable form
   *  @param sortedByFeature Whether to sort by feature names
   */
  public void justificationOf(Datum example, PrintWriter pw,
      Function printer, boolean sortedByFeature) {
    NumberFormat nf = NumberFormat.getNumberInstance();
    nf.setMinimumFractionDigits(2);
    nf.setMaximumFractionDigits(2);
    if (nf instanceof DecimalFormat) {
      ((DecimalFormat) nf).setPositivePrefix(" ");
    }

    // determine width for features, making it at least total's width
    int featureLength = 0;
    for (Object f : example.asFeatures()) {
      if (printer != null) {
        f = printer.apply(f);
      }
      featureLength = Math.max(featureLength, f.toString().length());
    }
    // make as wide as total printout
    featureLength = Math.max(featureLength, "Total:".length());
    // don't make it ridiculously wide
    featureLength = Math.min(featureLength, MAX_FEATURE_ALIGN_WIDTH);

    // determine width for labels
    int labelLength = 6;
    for (Object l : labels()) {
      labelLength = Math.max(labelLength, l.toString().length());
    }

    // print header row of output listing classes
    StringBuilder header = new StringBuilder("");
    for (int s = 0; s < featureLength; s++) {
      header.append(' ');
    }
    for (Object l : labels()) {
      header.append(' ');
      header.append(StringUtils.pad(l, labelLength));
    }
    pw.println(header);

    // print active features and weights per class
    Collection featColl = example.asFeatures();
    if (sortedByFeature) {
      List feats = new ArrayList(featColl);
      Collections.sort(feats);
      featColl = feats;
    }
    for (Object f : featColl) {
      String fStr;
      if (printer != null) {
        fStr = printer.apply(f).toString();
      } else {
        fStr = f.toString();
      }
      StringBuilder line = new StringBuilder(fStr);
      for (int s = fStr.length(); s < featureLength; s++) {
        line.append(' ');
      }
      for (Object l : labels()) {
        String lStr = nf.format(weight(f, l));
        line.append(' ');
        line.append(lStr);
        for (int s = lStr.length(); s < labelLength; s++) {
          line.append(' ');
        }
      }
      pw.println(line);
    }

    // Print totals, probs, etc.
    ClassicCounter scores = scoresOf(example);
    StringBuilder footer = new StringBuilder("Total:");
    for (int s = footer.length(); s < featureLength; s++) {
      footer.append(' ');
    }
    for (Object l : labels()) {
      footer.append(' ');
      String str = nf.format(scores.getCount(l));
      footer.append(str);
      for (int s = str.length(); s < labelLength; s++) {
        footer.append(' ');
      }
    }
    pw.println(footer);
    Distribution distr = Distribution.distributionFromLogisticCounter(scores);
    footer = new StringBuilder("Prob:");
    for (int s = footer.length(); s < featureLength; s++) {
      footer.append(' ');
    }
    for (Object l : labels()) {
      footer.append(' ');
      String str = nf.format(distr.getCount(l));
      footer.append(str);
      for (int s = str.length(); s < labelLength; s++) {
        footer.append(' ');
      }
    }
    pw.println(footer);
  }


  /**
   * Print all features active for a particular datum and the weight that
   * the classifier assigns to each class for those features.
   */
  public void justificationOf(Datum example, PrintWriter pw) {
    justificationOf(example, pw, null);
  }


  /**
   * Print all features in the classifier and the weight that they assign
   * to each class. The feature names are printed in sorted order.
   */
  public void dumpSorted() {
    Datum allFeatures = new BasicDatum(features(), "");
    justificationOf(allFeatures, new PrintWriter(System.err, true), true);
  }

  /**
   * Print all features active for a particular datum and the weight that
   * the classifier assigns to each class for those features. Sorts by feature
   * name if 'sorted' is true.
   */
  public void justificationOf(Datum example, PrintWriter pw, boolean sorted) {
    justificationOf(example, pw, null, sorted);
  }


  public ClassicCounter scoresOf(Datum example, Collection possibleLabels) {
    ClassicCounter scores = new ClassicCounter();
    for (Object l : possibleLabels) {
      if (labelIndex.indexOf(l) == -1) {
        continue;
      }
      double score = scoreOf(example, l);
      scores.setCount(l, score);
    }
    return scores;
  }


  public Object classOf(Datum example) {
    ClassicCounter scores = scoresOf(example);
    return Counters.argmax(scores);
  }

  public Object classOf(RVFDatum example) {
    ClassicCounter scores = scoresOf(example);
    return Counters.argmax(scores);
  }

  public LinearClassifier(double[][] weights, Index featureIndex, Index labelIndex) {
    this.featureIndex = featureIndex;
    this.labelIndex = labelIndex;
    this.weights = weights;
    thresholds = new double[labelIndex.size()];
    Arrays.fill(thresholds, 0.0);
  }

  public LinearClassifier(double[][] weights, Index featureIndex, Index labelIndex, 
      double[] thresholds) throws Exception {
    this.featureIndex = featureIndex;
    this.labelIndex = labelIndex;
    this.weights = weights;
    if (thresholds.length != labelIndex.size()) 
      throw new Exception("Number of thresholds and number of labels do not match.");
    thresholds = new double[thresholds.length];
    int curr = 0;
    for (double tval : thresholds) {
      thresholds[curr++] = tval;
    }
    Arrays.fill(thresholds, 0.0);
  }
  
  public LinearClassifier(ClassicCounter<Pair> weightCounter) {
    this(weightCounter, new ClassicCounter());
  }

  public LinearClassifier(ClassicCounter<Pair> weightCounter, ClassicCounter thresholdsC) {
    Collection<Pair> keys = weightCounter.keySet();
    featureIndex = new Index();
    labelIndex = new Index();
    for (Pair p : keys) {
      featureIndex.add(p.first);
      labelIndex.add(p.second);
    }
    thresholds = new double[labelIndex.size()];
    for (Object label : labelIndex) {
      thresholds[labelIndex.indexOf(label)] = thresholdsC.getCount(label);
    }
    weights = new double[featureIndex.size()][labelIndex.size()];
    Pair tempPair = new Pair();
    for (int f = 0; f < weights.length; f++) {
      for (int l = 0; l < weights[f].length; l++) {
        tempPair.first = featureIndex.get(f);
        tempPair.second = labelIndex.get(l);
        weights[f][l] = weightCounter.getCount(tempPair);
      }
    }
  }


  public void adaptWeights(Dataset adapt,LinearClassifierFactory lcf) {
    System.err.println("before adapting, weights size="+weights.length);
    weights = lcf.adaptWeights(weights,adapt);
    System.err.println("after adapting, weights size="+weights.length);
  }

  public double[][] weights() {
    return weights;
  }

  public void setWeights(double[][] newWeights) {
    weights = newWeights;
  }

  public static LinearClassifier readClassifier(String loadPath) {
    System.err.print("Deserializing classifier from " + loadPath + "...");

    try {

      ObjectInputStream ois;

      if (loadPath.endsWith("gz")) {
        ois = new ObjectInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(loadPath))));
      } else {
        ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(loadPath)));
      }

      LinearClassifier classifier = (LinearClassifier) ois.readObject();

      ois.close();
      return classifier;
    } catch (Exception e) {
      e.printStackTrace();
      throw new RuntimeException("Deserialization failed: "+e.getMessage());
    }
  }

  /**
   * Saves this out to a standard text file, instead of as a serialized Java object.
   * NOTE: this currently assumes feature and weights are represented as Strings.
   * @param file String filepath to write out to.
   */
  public void saveToFilename(String file) {
    try {
      File tgtFile = new File(file);
      BufferedWriter out = new BufferedWriter(new FileWriter(tgtFile));
      // output index first, blank delimiter, outline feature index, then weights
      labelIndex.saveToWriter(out);
      featureIndex.saveToWriter(out);
      int numLabels = labelIndex.size();
      int numFeatures = featureIndex.size();
      for (int featIndex=0; featIndex<numFeatures; featIndex++) {
        for (int labelIndex=0;labelIndex<numLabels;labelIndex++) {
          out.write(String.valueOf(featIndex));
          out.write(TEXT_SERIALIZATION_DELIMITER);
          out.write(String.valueOf(labelIndex));
          out.write(TEXT_SERIALIZATION_DELIMITER);
          out.write(String.valueOf(weight(featIndex, labelIndex)));
          out.write("\n");
        }
      }
      
      // write out thresholds: first item after blank is the number of thresholds, after is the threshold array values.
      out.write("\n");
      out.write(String.valueOf(thresholds.length));
      out.write("\n");
      for (double val : thresholds) {
        out.write(String.valueOf(val));
        out.write("\n");       
      }
      out.close();
    } catch (Exception e) {
      System.err.println("Error attempting to save classifier to file="+file);
      e.printStackTrace();
    }
  }


}
