// Stanford Classifier - a multiclass maxent classifier
// LogisticClassifier
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

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Properties;

import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.optimization.Minimizer;
import edu.stanford.nlp.optimization.QNMinimizer;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.util.FileLines;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.StringUtils;

/**
 * A classifier for binary logistic regression problems.
 * This uses the standard statistics textbook formulation of binary
 * logistic regression, which is more efficient than using the
 * LinearClassifier class.
 *
 * @author Galen Andrew
 */
public class LogisticClassifier implements Classifier, Serializable, RVFClassifier {
  private double[] weights;
  private Index featureIndex;
  private Object[] classes = new Object[2];
  private LogPrior prior;
  private boolean biased = false;

  public String toString() {
    StringBuffer sb = new StringBuffer();
    for (Object f : featureIndex) {
      System.err.println(classes[1]+" / "+f+" = "+weights[featureIndex.indexOf(f)]);
    }

    return sb.toString();
  }

  public Counter weightsAsCounter() {
    Counter c = new ClassicCounter();
    for (Object f : featureIndex) {
      c.incrementCount(classes[1]+" / "+f, weights[featureIndex.indexOf(f)]);
    }

    return c;
  }
  
  public LogisticClassifier() {
    this(new LogPrior(LogPrior.LogPriorType.QUADRATIC));
  }

  public LogisticClassifier(boolean biased) {
    this(new LogPrior(LogPrior.LogPriorType.QUADRATIC), biased);
  }
  
  public LogisticClassifier(LogPrior prior) {
    this.prior = prior;
  }

  
  public LogisticClassifier(LogPrior prior, boolean biased) {
    this.prior = prior;
    this.biased = biased;
  }

  public Collection labels() {
    Collection<Object> l = new LinkedList<Object>();
    l.add(classes[0]);
    l.add(classes[1]);
    return l;    
  }
  
  public Object classOf(Collection features) {
    if (scoreOf(features) > 0) {
      return classes[1];
    } else {
      return classes[0];
    }
  }


  public double scoreOf(Collection features) {
    double sum = 0;
    for (Iterator iterator = features.iterator(); iterator.hasNext();) {
      int f = featureIndex.indexOf(iterator.next());
      if (f >= 0) {
        sum += weights[f];
      }
    }
    return sum;
  }

  public ClassicCounter scoresOf(Datum datum) {
    Collection features = datum.asFeatures();
    double sum = scoreOf(features);
    ClassicCounter c = new ClassicCounter();
    c.setCount(classes[0], sum);
    c.setCount(classes[1], 1-sum);
    return c;
  }

  
  public Object classOf(Datum datum) {
    return classOf(datum.asFeatures());
  }
  
  public Object classOf(ClassicCounter features) {
    if (scoreOf(features) > 0) {
      return classes[1];
    } else {
      return classes[0];
    }
  }

  public double scoreOf(ClassicCounter features) {
    double sum = 0;
    for (Object feature : features.keySet()) {
      int f = featureIndex.indexOf(feature);
      if (f >= 0) {
        sum += weights[f]*features.getCount(feature);
      }
    }
    return sum;
  }
  
  public Object classOf(RVFDatum example) {
    return classOf(example.asFeaturesCounter());
  }

  public ClassicCounter scoresOf(RVFDatum example) {
    ClassicCounter features = example.asFeaturesCounter();
    double sum = scoreOf(features);
    ClassicCounter c = new ClassicCounter();
    System.out.println(classes[0] + ": " + sum +" ; " + classes[1] + ": " + (1-sum));
    c.setCount(classes[0], sum);
    c.setCount(classes[1], 1-sum);
    return c;
  }


  public double probabilityOf(Datum example) {
    return probabilityOf(example.asFeatures(), example.label());
  }

  public double probabilityOf(Collection features, Object label) {
    short sign = (short)(label.equals(classes[0]) ? 1 : -1);
    return 1.0 / (1.0 + Math.exp(sign * scoreOf(features)));
  }

  public double probabilityOf(RVFDatum example) {
    return probabilityOf(example.asFeaturesCounter(), example.label());
  }

  public double probabilityOf(ClassicCounter features, Object label) {
    short sign = (short)(label.equals(classes[0]) ? 1 : -1);
    return 1.0 / (1.0 + Math.exp(sign * scoreOf(features)));
  }
  

  public void train(GeneralDataset data) {
    if (data.labelIndex.size() != 2) {
      throw new RuntimeException("LogisticClassifier is only for binary classification!");
    }

    Minimizer minim;
    if (!biased) {
      LogisticObjectiveFunction lof = new LogisticObjectiveFunction(data.numFeatureTypes(), data.getDataArray(), data.getLabelsArray(), prior);
      minim = new QNMinimizer(lof);
      weights = minim.minimize(lof, 1e-4, new double[data.numFeatureTypes()]);
    } else {
      BiasedLogisticObjectiveFunction lof = new BiasedLogisticObjectiveFunction(data.numFeatureTypes(), data.getDataArray(), data.getLabelsArray(), prior);
      minim = new QNMinimizer(lof);
      weights = minim.minimize(lof, 1e-4, new double[data.numFeatureTypes()]);
    }

    featureIndex = data.featureIndex;
    classes[0] = data.labelIndex.get(0);
    classes[1] = data.labelIndex.get(1);
  }


  public static void main(String[] args) throws Exception {
    Properties prop = StringUtils.argsToProperties(args);

    Dataset ds = new Dataset();
    for (String line : new FileLines(prop.getProperty("trainFile"))) {
      String[] bits = line.split("\\s+");
      Collection f = new LinkedList();
      String l = bits[0];
      for (int i=1; i < bits.length; i++) {
        f.add(bits[i]);
      }
      ds.add(f, l);
    }

    ds.summaryStatistics();
    
    LogisticClassifier lc = new LogisticClassifier();
    if (prop.getProperty("biased", "false").equals("true")) {
      lc.biased = true;
    }
    lc.train(ds);


    for (String line : new FileLines(prop.getProperty("testFile"))) {
      String[] bits = line.split("\\s+");
      Collection f = new LinkedList();
      String l = bits[0];
      for (int i=1; i < bits.length; i++) {
        f.add(bits[i]);
      }
      String g = (String)lc.classOf(f);
      System.out.println(g+"\t"+line);
    }

  }


}
