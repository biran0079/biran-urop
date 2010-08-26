// Stanford Classifier - a multiclass maxent classifier
// NaiveBayesClassifierFactory
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

import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.optimization.HasInitial;
import edu.stanford.nlp.optimization.Minimizer;
import edu.stanford.nlp.optimization.QNMinimizer;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.Pair;
import java.util.*;

/**
 * @author Kristina Toutanova (kristina@cs.stanford.edu)
 *         creates a NaiveBayesClassifier given an RVFDataset
 */
public class NaiveBayesClassifierFactory implements ClassifierFactory {
  public static final int JL = 0;
  public static final int CL = 1;
  public static final int UCL = 2;
  int kind = JL;
  double alphaClass;
  double alphaFeature;
  double sigma;
  int prior = LogPrior.LogPriorType.NULL.ordinal();
  Index labelIndex;
  Index featureIndex;

  public NaiveBayesClassifierFactory() {
  }

  public NaiveBayesClassifierFactory(double alphaC, double alphaF, double sigma, int prior, int kind) {
    alphaClass = alphaC;
    alphaFeature = alphaF;
    this.sigma = sigma;
    this.prior = prior;
    this.kind = kind;
  }

  private NaiveBayesClassifier trainClassifier(int[][] data, int[] labels, int numFeatures, int numClasses, Index labelIndex, Index featureIndex) {
    Set labelSet = new HashSet();
    NBWeights nbWeights = trainWeights(data, labels, numFeatures, numClasses);
    ClassicCounter priors = new ClassicCounter();
    double[] pr = nbWeights.priors;
    for (int i = 0; i < pr.length; i++) {
      priors.incrementCount(labelIndex.get(i), pr[i]);
      labelSet.add(labelIndex.get(i));
    }
    ClassicCounter weightsCounter = new ClassicCounter();
    double[][][] wts = nbWeights.weights;
    for (int c = 0; c < numClasses; c++) {
      Object label = labelIndex.get(c);
      for (int f = 0; f < numFeatures; f++) {
        Object feature = featureIndex.get(f);
        Pair p = new Pair(label, feature);
        for (int val = 0; val < wts[c][f].length; val++) {
          Pair key = new Pair(p, new Integer(val));
          weightsCounter.incrementCount(key, wts[c][f][val]);
        }
      }
    }
    return new NaiveBayesClassifier(weightsCounter, priors, labelSet);

  }

  /**
   * The examples are assumed a list of RFVDatum
   * the datums are assumed to contain the zeros as well
   *
   * @param examples
   * @return
   */
  public Classifier trainClassifier(List examples) {
    RVFDatum d0 = (RVFDatum) examples.get(0);
    int numFeatures = d0.asFeatures().size();
    int[][] data = new int[examples.size()][numFeatures];
    int[] labels = new int[examples.size()];
    labelIndex = new Index();
    featureIndex = new Index();
    for (int d = 0; d < examples.size(); d++) {
      RVFDatum datum = (RVFDatum) examples.get(d);
      ClassicCounter c = datum.asFeaturesCounter();
      for (Iterator it = c.keySet().iterator(); it.hasNext();) {
        Object feature = it.next();
        featureIndex.add(feature);
        int fNo = featureIndex.indexOf(feature);
        int value = (int) c.getCount(feature);
        data[d][fNo] = value;
      }
      labelIndex.add(datum.label());
      labels[d] = labelIndex.indexOf(datum.label());

    }
    int numClasses = labelIndex.size();
    return trainClassifier(data, labels, numFeatures, numClasses, labelIndex, featureIndex);
  }


  /**
   * The examples are assumed a list of RFVDatum
   * the datums are assumed to not contain the zeros and then they are added to each instance
   *
   * @param examples
   * @return
   */
  public Classifier trainClassifier(List examples, Set featureSet) {
    int numFeatures = featureSet.size();
    int[][] data = new int[examples.size()][numFeatures];
    int[] labels = new int[examples.size()];
    labelIndex = new Index();
    featureIndex = new Index();
    for (Iterator fi = featureSet.iterator(); fi.hasNext();) {
      featureIndex.add(fi.next());
    }
    for (int d = 0; d < examples.size(); d++) {
      RVFDatum datum = (RVFDatum) examples.get(d);
      ClassicCounter c = datum.asFeaturesCounter();
      for (Iterator it = c.keySet().iterator(); it.hasNext();) {
        Object feature = it.next();
        int fNo = featureIndex.indexOf(feature);
        int value = (int) c.getCount(feature);
        data[d][fNo] = value;
      }
      labelIndex.add(datum.label());
      labels[d] = labelIndex.indexOf(datum.label());

    }
    int numClasses = labelIndex.size();
    return trainClassifier(data, labels, numFeatures, numClasses, labelIndex, featureIndex);
  }


  /**
   * Here the data is assumed to be for every instance, array of length numFeatures and the value of the feature is stored including zeros
   *
   * @param data
   * @param labels
   * @return label,fno,value -> weight
   */
  private NBWeights trainWeights(int[][] data, int[] labels, int numFeatures, int numClasses) {
    if (kind == JL) {
      return trainWeightsJL(data, labels, numFeatures, numClasses);
    }
    if (kind == UCL) {
      return trainWeightsUCL(data, labels, numFeatures, numClasses);
    }
    if (kind == CL) {
      return trainWeightsCL(data, labels, numFeatures, numClasses);
    }
    return null;
  }

  private NBWeights trainWeightsJL(int[][] data, int[] labels, int numFeatures, int numClasses) {
    int[] numValues = numberValues(data, numFeatures);
    double[] priors = new double[numClasses];
    double[][][] weights = new double[numClasses][numFeatures][];
    //init weights array
    for (int cl = 0; cl < numClasses; cl++) {
      for (int fno = 0; fno < numFeatures; fno++) {
        weights[cl][fno] = new double[numValues[fno]];
      }
    }
    for (int i = 0; i < data.length; i++) {
      priors[labels[i]]++;
      for (int fno = 0; fno < numFeatures; fno++) {
        weights[labels[i]][fno][data[i][fno]]++;
      }
    }
    for (int cl = 0; cl < numClasses; cl++) {
      for (int fno = 0; fno < numFeatures; fno++) {
        for (int val = 0; val < numValues[fno]; val++) {
          weights[cl][fno][val] = Math.log((weights[cl][fno][val] + alphaFeature) / (priors[cl] + alphaFeature * numValues[fno]));
        }
      }
      priors[cl] = Math.log((priors[cl] + alphaClass) / (data.length + alphaClass * numClasses));
    }
    return new NBWeights(priors, weights);
  }

  private NBWeights trainWeightsUCL(int[][] data, int[] labels, int numFeatures, int numClasses) {
    int[] numValues = numberValues(data, numFeatures);
    int[] sumValues = new int[numFeatures]; //how many feature-values are before this feature
    for (int j = 1; j < numFeatures; j++) {
      sumValues[j] = sumValues[j - 1] + numValues[j - 1];
    }
    int[][] newdata = new int[data.length][numFeatures + 1];
    for (int i = 0; i < data.length; i++) {
      newdata[i][0] = 0;
      for (int j = 0; j < numFeatures; j++) {
        newdata[i][j + 1] = sumValues[j] + data[i][j] + 1;
      }
    }
    int totalFeatures = sumValues[numFeatures - 1] + numValues[numFeatures - 1] + 1;
    System.err.println("total feats " + totalFeatures);
    LogConditionalObjectiveFunction objective = new LogConditionalObjectiveFunction(totalFeatures, numClasses, newdata, labels, prior, sigma, 0.0);
    Minimizer min = new QNMinimizer();
    double[] argmin = min.minimize(objective, 1e-4, ((HasInitial) objective).initial());
    double[][] wts = objective.to2D(argmin);
    System.out.println("weights have dimension " + wts.length);
    return new NBWeights(wts, numValues);
  }


  private NBWeights trainWeightsCL(int[][] data, int[] labels, int numFeatures, int numClasses) {

    LogConditionalEqConstraintFunction objective = new LogConditionalEqConstraintFunction(numFeatures, numClasses, data, labels, prior, sigma, 0.0);
    Minimizer min = new QNMinimizer();
    double[] argmin = min.minimize(objective, 1e-4, ((HasInitial) objective).initial());
    double[][][] wts = objective.to3D(argmin);
    double[] priors = objective.priors(argmin);
    return new NBWeights(priors, wts);
  }

  static int[] numberValues(int[][] data, int numFeatures) {
    int[] numValues = new int[numFeatures];
    for (int i = 0; i < data.length; i++) {
      for (int j = 0; j < data[i].length; j++) {
        if (numValues[j] < data[i][j] + 1) {
          numValues[j] = data[i][j] + 1;
        }
      }
    }
    return numValues;
  }

  class NBWeights {
    double[] priors;
    double[][][] weights;

    NBWeights(double[] priors, double[][][] weights) {
      this.priors = priors;
      this.weights = weights;
    }

    /**
     * create the parameters from a coded representation
     * where feature 0 is the prior etc.
     *
     * @param wts
     * @param numValues
     */
    NBWeights(double[][] wts, int[] numValues) {
      int numClasses = wts[0].length;
      priors = new double[numClasses];
      for (int j = 0; j < numClasses; j++) {
        priors[j] = wts[0][j];
      }
      int[] sumValues = new int[numValues.length];
      for (int j = 1; j < numValues.length; j++) {
        sumValues[j] = sumValues[j - 1] + numValues[j - 1];
      }
      weights = new double[priors.length][sumValues.length][];
      for (int fno = 0; fno < numValues.length; fno++) {
        for (int c = 0; c < numClasses; c++) {
          weights[c][fno] = new double[numValues[fno]];
        }

        for (int val = 0; val < numValues[fno]; val++) {
          int code = sumValues[fno] + val + 1;
          for (int cls = 0; cls < numClasses; cls++) {
            weights[cls][fno][val] = wts[code][cls];
          }
        }
      }
    }
  }

  public static void main(String[] args) {

    /*
    List examples = new ArrayList();
    String leftLight = "leftLight";
    String rightLight = "rightLight";
    String broken = "BROKEN";
    String ok = "OK";
    Counter c1 = new Counter();
    c1.incrementCount(leftLight, 0);
    c1.incrementCount(rightLight, 0);
    RVFDatum d1 = new RVFDatum(c1, broken);
    examples.add(d1);
    Counter c2 = new Counter();
    c2.incrementCount(leftLight, 1);
    c2.incrementCount(rightLight, 1);
    RVFDatum d2 = new RVFDatum(c2, ok);
    examples.add(d2);
    Counter c3 = new Counter();
    c3.incrementCount(leftLight, 0);
    c3.incrementCount(rightLight, 1);
    RVFDatum d3 = new RVFDatum(c3, ok);
    examples.add(d3);
    Counter c4 = new Counter();
    c4.incrementCount(leftLight, 1);
    c4.incrementCount(rightLight, 0);
    RVFDatum d4 = new RVFDatum(c4, ok);
    examples.add(d4);
    NaiveBayesClassifier classifier = (NaiveBayesClassifier) new NaiveBayesClassifierFactory(200, 200, 1.0, LogPrior.QUADRATIC.ordinal(), NaiveBayesClassifierFactory.CL).trainClassifier(examples);
    classifier.print();
    //now classifiy
    for (int i = 0; i < examples.size(); i++) {
        RVFDatum d = (RVFDatum) examples.get(i);
        Counter scores = classifier.scoresOf(d);
        System.out.println("for datum " + d + " scores are " + scores.toString());
        System.out.println(" class is " + scores.argmax());
    }

}
*/
    String trainFile = args[0];
    String testFile = args[1];
    NominalDataReader nR = new NominalDataReader();
    HashMap indices = new HashMap();
    List train = nR.readData(trainFile, indices);
    List test = nR.readData(testFile, indices);
    System.out.println("Constrained conditional likelihood no prior :");
    for (int j = 0; j < 100; j++) {
      NaiveBayesClassifier classifier = (NaiveBayesClassifier) new NaiveBayesClassifierFactory(0.1, 0.01, 0.6, LogPrior.LogPriorType.NULL.ordinal(), NaiveBayesClassifierFactory.CL).trainClassifier(train);
      classifier.print();
      //now classifiy

      float accTrain = classifier.accuracy(train.iterator());
      System.err.println("training accuracy " + accTrain);
      float accTest = classifier.accuracy(test.iterator());
      System.err.println("test accuracy " + accTest);

    }
    System.out.println("Unconstrained conditional likelihood no prior :");
    for (int j = 0; j < 100; j++) {
      NaiveBayesClassifier classifier = (NaiveBayesClassifier) new NaiveBayesClassifierFactory(0.1, 0.01, 0.6, LogPrior.LogPriorType.NULL.ordinal(), NaiveBayesClassifierFactory.UCL).trainClassifier(train);
      classifier.print();
      //now classifiy

      float accTrain = classifier.accuracy(train.iterator());
      System.err.println("training accuracy " + accTrain);
      float accTest = classifier.accuracy(test.iterator());
      System.err.println("test accuracy " + accTest);

    }
  }
}
