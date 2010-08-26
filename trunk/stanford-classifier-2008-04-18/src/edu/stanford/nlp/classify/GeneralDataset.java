package edu.stanford.nlp.classify;

import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.stats.ClassicCounter;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.io.PrintWriter;

/**
 * The purpose of this interface is to unify {@link Dataset} and {@link RVFDataset}.
 * @author Kristina Toutanova (kristina@cs.stanford.edu)
 * @author Anna Rafferty (various refactoring with subclasses)
 */
public abstract class GeneralDataset {

  public Index labelIndex;
  public Index featureIndex;

  protected int[] labels;
  protected int[][] data;

  protected int size;

  public GeneralDataset() {
  }

  public Index labelIndex() { return labelIndex; }

  public Index featureIndex() { return featureIndex; }

  public int numFeatures() { return featureIndex.size(); }

  public int numClasses() { return labelIndex.size(); }

  public int[] getLabelsArray() {
    labels = trimToSize(labels);
    return labels;
  }

  public int[][] getDataArray() {
    data = trimToSize(data);
    return data;
  }
  
  public abstract double[][] getValuesArray();
  
  /**
   * Resets the Dataset so that it is empty and ready to collect data.
   */
  public void clear() {
    clear(10);
  }

  /**
   * Resets the Dataset so that it is empty and ready to collect data.
   * @param numDatums initial capacity of dataset
   */
  public void clear(int numDatums) {
    initialize(numDatums);
  }
  
  /**
   * This method takes care of resetting values of the dataset
   * such that it is empty with an initial capacity of numDatums
   * 
   * Should be accessed only by appropriate methods within the class,
   * such as clear(), which take care of other parts of the emptying of data
   * 
   * @param numDatums initial capacity of dataset
   */
  protected abstract void initialize(int numDatums);

  public abstract RVFDatum getRVFDatum(int index);

  public abstract void add(Datum d);
  
  /**
   * Get the total count (over all data instances) of each feature
   *
   * @return an array containing the counts (indexed by index)
   */
  protected float[] getFeatureCounts() {
    float[] counts = new float[featureIndex.size()];
    for (int i = 0, m = size; i < m; i++) {
      for (int j = 0, n = data[i].length; j < n; j++) {
        counts[data[i][j]] += 1.0;
      }
    }
    return counts;
  }
  
  /**
   * Applies a feature count threshold to the Dataset.  All features that
   * occur fewer than <i>k</i> times are expunged.
   */
  public void applyFeatureCountThreshold(int k) {
    float[] counts = getFeatureCounts();
    Index<Object> newFeatureIndex = new Index<Object>();

    int[] featMap = new int[featureIndex.size()];
    for (int i = 0; i < featMap.length; i++) {
      Object feat = featureIndex.get(i);
      if (counts[i] >= k) {
        int newIndex = newFeatureIndex.size();
        newFeatureIndex.add(feat);
        featMap[i] = newIndex;
      } else {
        featMap[i] = -1;
      }
      featureIndex.remove(feat);
    }

    featureIndex = newFeatureIndex;
    counts = null;

    for (int i = 0; i < size; i++) {
      List<Integer> featList = new ArrayList<Integer>(data[i].length);
      for (int j = 0; j < data[i].length; j++) {
        if (featMap[data[i][j]] >= 0) {
          featList.add(featMap[data[i][j]]);
        }
      }
      data[i] = new int[featList.size()];
      for (int j = 0; j < data[i].length; j++) {
        data[i][j] = featList.get(j);
      }
    }
    
  }
  
  /**
   * returns the number of feature tokens in the Dataset.
   */
  public int numFeatureTokens() {
    int x = 0;
    for (int i = 0, m = size; i < m; i++) {
      x += data[i].length;
    }
    return x;
  }

  /**
   * returns the number of distinct feature types in the Dataset.
   */
  public int numFeatureTypes() {
    return featureIndex.size();
  }
  


  /**
   * Adds all Datums in the given collection of data to this dataset
   * @param data collection of datums you would like to add to the dataset
   */
  public void addAll(Collection<Datum> data) {
    for (Datum d : data) {
      add(d);
    }
  }

  public abstract Pair<GeneralDataset, GeneralDataset> split (int start, int end) ;
  public abstract Pair<GeneralDataset, GeneralDataset> split (double p) ;

  /**
   * Returns the number of examples ({@link Datum}s) in the Dataset.
   */
  public int size() { return size; }

  protected void trimData() {
    data = trimToSize(data);
  }

  protected void trimLabels() {
    labels = trimToSize(labels);
  }

  protected int[] trimToSize(int[] i) {
    int[] newI = new int[size];
    System.arraycopy(i, 0, newI, 0, size);
    return newI;
  }

  protected int[][] trimToSize(int[][] i) {
    int[][] newI = new int[size][];
    System.arraycopy(i, 0, newI, 0, size);
    return newI;
  }

  protected double[][] trimToSize(double[][] i) {
    double[][] newI = new double[size][];
    System.arraycopy(i, 0, newI, 0, size);
    return newI;
  }
  

  /**
   * Print some statistics summarizing the dataset
   *
   */
  public abstract void summaryStatistics();
  
  /**
   * Returns an iterator over the class labels of the Dataset
   */
  public Iterator labelIterator() {
    return labelIndex.iterator();
  }


  /**
   * Dumps the Dataset as a training/test file for SVMLight. <br>
   * class [fno:val]+
   * The features must occur in consecutive order.
   */
  public void printSVMLightFormat() {
    printSVMLightFormat(new PrintWriter(System.out));
  }

  /**
   * Print SVM Light Format file.  If the Dataset has more than 2 classes, then it
   * prints using the label index (+1) (for svm_struct).  If it is 2 classes, then the labelIndex.get(0)
   * is mapped to +1 and labelIndex.get(1) is mapped to -1 (for svm_light).
   */
  public void printSVMLightFormat(PrintWriter pw) {
    //assumes each data item has a few features on, and sorts the feature keys while collecting the values in a counter
    String[] labelMap = new String[numClasses()];
    if (numClasses() > 2) {
      for (int i = 0; i < labelMap.length; i++) {
        labelMap[i] = String.valueOf((i + 1));
      }
    } else {
      labelMap = new String[]{"+1", "-1"};
    }

    for (int i = 0; i < size; i++) {
      RVFDatum d = getRVFDatum(i);
      ClassicCounter<Object> c = d.asFeaturesCounter();
      ClassicCounter<Integer> printC = new ClassicCounter<Integer>();
      for (Object f : c.keySet()) {
        printC.setCount(featureIndex.indexOf(f), c.getCount(f));
      }
      Integer[] features = printC.keySet().toArray(new Integer[0]);
      Arrays.sort(features);
      StringBuilder sb = new StringBuilder();
      sb.append(labelMap[labels[i]]).append(" ");
      for (int f: features) {
        sb.append((f + 1)).append(":").append(c.getCount(f)).append(" ");
      }
      pw.println(sb.toString());
    }
  }

}
