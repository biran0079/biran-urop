// Stanford Classifier - a multiclass maxent classifier
// NaiveBayesClassifier
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

import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.Pair;
import java.io.PrintStream;
import java.util.Iterator;
import java.util.Set;
import java.util.Collection;

/**
 * @author Kristina Toutanova (kristina@cs.stanford.edu)
 *         A Naive Bayes classifier with a fixed number of features.
 *         The features are assumed to have integer values even though RVFDatum will return doubles
 */
public class NaiveBayesClassifier implements Classifier, RVFClassifier {
  ClassicCounter weights; //the keys will be class and feature and value
  ClassicCounter priors;
  Set features; // we need all features to add the weights for zero-valued ones
  private boolean addZeroValued; // whether to add features as having value 0 if they are not in Datum/RFVDatum
  ClassicCounter priorZero; //if we need to add the zeros, pre-compute the weight for all zeros for each class
  Set labels;
  private final Integer zero = new Integer(0);

  public Collection labels() {
    return labels;
  }

  public Object classOf(RVFDatum example) {
    ClassicCounter scores = scoresOf(example);
    return Counters.argmax(scores);
  }

  public ClassicCounter scoresOf(RVFDatum example) {
    ClassicCounter scores = new ClassicCounter();
    Counters.addInPlace(scores, priors);
    if (addZeroValued) {
      Counters.addInPlace(scores, priorZero);
    }
    for (Iterator it = labels.iterator(); it.hasNext();) {
      Object l = it.next();
      double score = 0.0;
      ClassicCounter features = example.asFeaturesCounter();
      for (Iterator j = features.keySet().iterator(); j.hasNext();) {
        Object f = j.next();
        int value = (int) features.getCount(f);
        score += weight(l, f, new Integer(value));
        if (addZeroValued) {
          score -= weight(l, f, zero);
        }
      }
      scores.incrementCount(l, score);
    }
    return scores;
  }


  public Object classOf(Datum example) {
    RVFDatum rvf = new RVFDatum(example);
    return classOf(rvf);
  }

  public ClassicCounter scoresOf(Datum example) {
    RVFDatum rvf = new RVFDatum(example);
    return scoresOf(rvf);
  }

  public NaiveBayesClassifier(ClassicCounter weights, ClassicCounter priors, Set labels, Set features, boolean addZero) {
    this.weights = weights;
    this.features = features;
    this.priors = priors;
    this.labels = labels;
    addZeroValued = addZero;
    if (addZeroValued) {
      initZeros();
    }
  }


  public float accuracy(Iterator exampleIterator) {
    int correct = 0;
    int total = 0;
    for (; exampleIterator.hasNext();) {
      RVFDatum next = (RVFDatum) exampleIterator.next();
      Object guess = classOf(next);
      if (guess.equals(next.label())) {
        correct++;
      }
      total++;
    }
    System.err.println("correct " + correct + " out of " + total);
    return correct / (float) total;
  }

  public void print(PrintStream pw) {
    pw.println("priors ");
    pw.println(priors.toString());
    pw.println("weights ");
    pw.println(weights.toString());
  }

  public void print() {
    print(System.out);
  }

  private double weight(Object label, Object feature, Object val) {
    Pair p = new Pair(new Pair(label, feature), val);
    double v = weights.getCount(p);
    return v;
  }

  public NaiveBayesClassifier(ClassicCounter weights, ClassicCounter priors, Set labels) {
    this(weights, priors, labels, null, false);
  }

  /**
   * In case the features for which there is a value 0 in an example need to have their coefficients multiplied in,
   * we need to pre-compute the addition
   * priorZero(l)=sum_{features} wt(l,feat=0)
   */
  private void initZeros() {
    priorZero = new ClassicCounter();
    for (Iterator it = labels.iterator(); it.hasNext();) {
      Object label = it.next();
      double score = 0;
      for (Iterator featIter = features.iterator(); featIter.hasNext();) {
        Object feature = featIter.next();
        score += weight(label, feature, zero);
      }
      priorZero.setCount(label, score);
    }
  }


}
