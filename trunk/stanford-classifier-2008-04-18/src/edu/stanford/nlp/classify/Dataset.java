package edu.stanford.nlp.classify;

import java.io.BufferedReader;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.GeneralizedCounter;
import edu.stanford.nlp.util.FileLines;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.ScoredComparator;
import edu.stanford.nlp.util.ScoredObject;


/**
 * An interfacing class for {@link ClassifierFactory} that incrementally
 * builds a more memory-efficent representation of a {@link List} of
 * {@link Datum} objects for the purposes of training a {@link Classifier}
 * with a {@link ClassifierFactory}.
 *
 * @author Roger Levy (rog@stanford.edu)
 * @author Anna Rafferty (various refactoring with GeneralDataset/RVFDataset)
 */
public class Dataset extends GeneralDataset {

  public Dataset() {
    this(10);
  }


  public Dataset(int numDatums, Index featureIndex, Index labelIndex) {
    initialize(numDatums);
    this.labelIndex = labelIndex;
    this.featureIndex = featureIndex;
  }

  public Dataset(int numDatums) {
    initialize(numDatums);
  }

  /**
   * Constructor that fully specifies a Dataset.  Needed this for MulticlassDataset.
   */
  public Dataset(Index labelIndex, int[] labels, Index featureIndex, int[][] data) {
    this (labelIndex, labels, featureIndex, data, data.length);
  }

  /**
   * Constructor that fully specifies a Dataset.  Needed this for MulticlassDataset.
   */
  public Dataset(Index labelIndex, int[] labels, Index featureIndex, int[][] data, int size) {
    this.labelIndex = labelIndex;
    this.labels = labels;
    this.featureIndex = featureIndex;
    this.data = data;
    this.size = size;
  }

  public Pair<GeneralDataset, GeneralDataset> split(double percentDev) {
    int devSize = (int)(percentDev * size());
    int trainSize = size() - devSize;
    
    int[][] devData = new int[devSize][];
    int[] devLabels = new int[devSize];

    int[][] trainData = new int[trainSize][];
    int[] trainLabels = new int[trainSize];

    System.arraycopy(data, 0, devData, 0, devSize);
    System.arraycopy(labels, 0, devLabels, 0, devSize);

    System.arraycopy(data, devSize, trainData, 0, trainSize);
    System.arraycopy(labels, devSize, trainLabels, 0, trainSize);

    if (this instanceof WeightedDataset) {
      float[] trainWeights = new float[trainSize];
      float[] devWeights = new float[devSize];

      WeightedDataset w = (WeightedDataset)this;

      System.arraycopy(w.weights, 0, devWeights, 0, devSize);
      System.arraycopy(w.weights, devSize, trainWeights, 0, trainSize);
      
      WeightedDataset dev = new WeightedDataset(labelIndex, devLabels, featureIndex, devData, devSize, devWeights);
      WeightedDataset train = new WeightedDataset(labelIndex, trainLabels, featureIndex, trainData, trainSize, trainWeights);

      return new Pair<GeneralDataset,GeneralDataset>(train, dev);
    } else {
      
      Dataset dev = new Dataset(labelIndex, devLabels, featureIndex, devData, devSize);
      Dataset train = new Dataset(labelIndex, trainLabels, featureIndex, trainData, trainSize);
      
      return new Pair<GeneralDataset,GeneralDataset>(train, dev);
    }
  }

  public Pair<GeneralDataset,GeneralDataset> split(int start, int end) {
    int devSize = end - start;
    int trainSize = size() - devSize;
    
    int[][] devData = new int[devSize][];
    int[] devLabels = new int[devSize];

    int[][] trainData = new int[trainSize][];
    int[] trainLabels = new int[trainSize];

    System.arraycopy(data, start, devData, 0, devSize);
    System.arraycopy(labels, start, devLabels, 0, devSize);

    System.arraycopy(data, 0, trainData, 0, start);
    System.arraycopy(data, end, trainData, start, size()-end);
    System.arraycopy(labels, 0, trainLabels, 0, start);
    System.arraycopy(labels, end, trainLabels, start, size()-end);

    if (this instanceof WeightedDataset) {
      float[] trainWeights = new float[trainSize];
      float[] devWeights = new float[devSize];

      WeightedDataset w = (WeightedDataset)this;

      System.arraycopy(w.weights, start, devWeights, 0, devSize);
      System.arraycopy(w.weights, 0, trainWeights, 0, start);
      System.arraycopy(w.weights, end, trainWeights, start, size()-end);
      
      WeightedDataset dev = new WeightedDataset(labelIndex, devLabels, featureIndex, devData, devSize, devWeights);
      WeightedDataset train = new WeightedDataset(labelIndex, trainLabels, featureIndex, trainData, trainSize, trainWeights);

      return new Pair<GeneralDataset,GeneralDataset>(train, dev);
    } else {
      
      Dataset dev = new Dataset(labelIndex, devLabels, featureIndex, devData, devSize);
      Dataset train = new Dataset(labelIndex, trainLabels, featureIndex, trainData, trainSize);
      
      return new Pair<GeneralDataset,GeneralDataset>(train, dev);
    }
  }


  public Dataset getRandomSubDataset(double p, int seed) {
    int newSize = (int)(p * size());
    Set<Integer> indicesToKeep = new HashSet<Integer>();
    Random r = new Random();
    int s = size();
    while (indicesToKeep.size() < newSize) {
      indicesToKeep.add(r.nextInt(s));
    }

    int[][] newData = new int[newSize][];
    int[] newLabels = new int[newSize];

    int i = 0;
    for (int j : indicesToKeep) {
      newData[i] = data[j];
      newLabels[i] = labels[j];
      i++;
    }

    return new Dataset(labelIndex, newLabels, featureIndex, newData);
  }

  public double[][] getValuesArray() {
    return null;
  }

  /**
   * Constructs a Dataset by reading in a file in SVM light format.
   */
  public static Dataset readSVMLightFormat(String filename) {
    return readSVMLightFormat(filename, new Index(), new Index());
  }
  
  /**
   * Constructs a Dataset by reading in a file in SVM light format.
   * The lines parameter is filled with the lines of the file for further processing
   * (if lines is null, it is assumed no line information is desired)
   */
  public static Dataset readSVMLightFormat(String filename, List<String> lines) {
    return readSVMLightFormat(filename, new Index(), new Index(), lines);
  }
  
  /**
   * Constructs a Dataset by reading in a file in SVM light format.
   * the created dataset has the same feature and label index as given
   */
  public static Dataset readSVMLightFormat(String filename, Index featureIndex, Index labelIndex) {
    return readSVMLightFormat(filename, featureIndex, labelIndex, null);
  }
  /**
   * Constructs a Dataset by reading in a file in SVM light format.
   * the created dataset has the same feature and label index as given
   */
  public static Dataset readSVMLightFormat(String filename, Index featureIndex, Index labelIndex, List<String> lines) {
    BufferedReader in = null;
    Dataset dataset;
    try {
      dataset = new Dataset(10, featureIndex, labelIndex);
      for (String line : new FileLines(filename)) {
        if(lines != null)
          lines.add(line);
        dataset.add(svmLightLineToDatum(line));
      }
      
    } catch (Exception e) {
      e.printStackTrace();
      throw new RuntimeException();
    } finally {
      if (in != null) {
        try {
          in.close();
        } catch (Exception ioe) {
        }
      }
    }
    return dataset;
  }

  private static int line1 = 0;

  public static Datum svmLightLineToDatum(String l) {
    line1++;
    String[] line = l.split("\\s+");
    List<Object> features = new ArrayList<Object>();
    for (int i = 1; i < line.length; i++) {
      String[] f = line[i].split(":");
      if (f.length != 2) { 
        System.err.println("Dataset error: line " + line1);
      }
      int val = (int) Double.parseDouble(f[1]);
      for (int j = 0; j < val; j++) {
        features.add(new Integer(f[0]));
      }
    }
    features.add("###");
    Datum d = new BasicDatum(features, line[0]);
    return d;
  }

  /**
   *  Get Number of datums a given feature appears in.
   */
  public ClassicCounter getFeatureCounter()
  {
    ClassicCounter<Object> featureCounts = new ClassicCounter<Object>();
    for (int i=0; i < data.length; ++i)
    {
      BasicDatum datum = (BasicDatum) getDatum(i);
      Set<Object> featureSet   = new HashSet<Object>(datum.asFeatures());
      for (Object key : featureSet) {
        featureCounts.incrementCount(key, 1.0);
      }
    }
    return featureCounts;
  }

  public void add(Datum d) {
    add(d.asFeatures(), d.label());
  }

  public void add(Collection features, Object label) {
    ensureSize();
    addLabel(label);
    addFeatures(features);
    size++;
  }

  protected void ensureSize() {
    if (labels.length == size) {
      int[] newLabels = new int[size * 2];
      System.arraycopy(labels, 0, newLabels, 0, size);
      labels = newLabels;
      int[][] newData = new int[size * 2][];
      System.arraycopy(data, 0, newData, 0, size);
      data = newData;
    }
  }

  protected void addLabel(Object label) {
    labelIndex.add(label);
    labels[size] = labelIndex.indexOf(label);
  }

  protected void addFeatures(Collection features) {
    int[] intFeatures = new int[features.size()];
    int j = 0;
    for (Iterator i = features.iterator(); i.hasNext();) {
      Object feature = i.next();
      featureIndex.add(feature);
      int index = featureIndex.indexOf(feature);
      if (index >= 0) {
        intFeatures[j] = featureIndex.indexOf(feature);
        j++;
      }
    }
    data[size] = new int[j];
    System.arraycopy(intFeatures, 0, data[size], 0, j);
  }



  protected void initialize(int numDatums) {
    labelIndex = new Index<Object>();
    featureIndex = new Index<Object>();
    labels = new int[numDatums];
    data = new int[numDatums][];
    size = 0;
  }

  /**
   * @param index
   * @return the index-ed datum
   */
  public Datum getDatum(int index) {
    return new BasicDatum(featureIndex.objects(data[index]), labelIndex.get(labels[index]));
  }

  /**
   * @param index
   * @return the index-ed datum
   */
  public RVFDatum getRVFDatum(int index) {
    ClassicCounter<Object> c = new ClassicCounter<Object>();
    for (Object key : featureIndex.objects(data[index])) {
      c.incrementCount(key, 1.0);
    }
    return new RVFDatum(c, labelIndex.get(labels[index]));
  }

  /**
   * Prints some summary statistics to stderr for the Dataset.
   */
  public void summaryStatistics() {
    System.err.println(toSummaryStatistics());
  }

  public String toSummaryStatistics() {
    StringBuilder sb = new StringBuilder();
    sb.append("numDatums: ").append(size).append("\n");
    sb.append("numLabels: ").append(labelIndex.size()).append(" [");
    Iterator iter = labelIndex.iterator();
    while (iter.hasNext()) {
      sb.append(iter.next());
      if (iter.hasNext()) {
        sb.append(", ");
      }
    }
    sb.append("]\n");
    sb.append("numFeatures Phi(X) types): ").append(featureIndex.size()).append("\n");
    // List l = new ArrayList(featureIndex);
//     Collections.sort(l);
//     sb.append(l);
    return sb.toString();
  }


  /**
   * Applies feature count thresholds to the Dataset.
   * Only features that match pattern_i and occur at
   * least threshold_i times (for some i) are kept.
   *
   * @param thresholds a list of pattern, threshold pairs
   */
  public void applyFeatureCountThreshold(List<Pair<Pattern, Integer>> thresholds) {

    // get feature counts
    float[] counts = getFeatureCounts();

    // build a new featureIndex
    Index<Object> newFeatureIndex = new Index<Object>();
    Iterator iter = featureIndex.iterator();
    LOOP: while (iter.hasNext()) {
      String f = (String) iter.next();
      Iterator<Pair<Pattern, Integer>> l = thresholds.iterator();
      while (l.hasNext()) {
        Pair<Pattern, Integer> pair = l.next();
        Pattern p = pair.first();
        Matcher m = p.matcher(f);
        if (m.matches()) {
          if (counts[featureIndex.indexOf(f)] >= pair.second) {
            newFeatureIndex.add(f);
          }
          continue LOOP;
        }
      }
      // we only get here if it didn't match anything on the list
      newFeatureIndex.add(f);
    }

    counts = null;

    int[] featMap = new int[featureIndex.size()];
    for (int i = 0; i < featMap.length; i++) {
      featMap[i] = newFeatureIndex.indexOf(featureIndex.get(i));
    }

    featureIndex = null;

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

    featureIndex = newFeatureIndex;
  }


  /**
   * prints the full feature matrix in tab-delimited form.  These can be BIG
   * matrices, so be careful!
   */
  public void printFullFeatureMatrix(PrintWriter pw) {
    String sep = "\t";
    for (int i = 0; i < featureIndex.size(); i++) {
      pw.print(sep + featureIndex.get(i));
    }
    pw.println();
    for (int i = 0; i < labels.length; i++) {
      pw.print(labelIndex.get(i));
      Set<Integer> feats = new HashSet<Integer>();
      for (int j = 0; j < data[i].length; j++) {
        int feature = data[i][j];
        feats.add(new Integer(feature));
      }
      for (int j = 0; j < featureIndex.size(); j++) {
        if (feats.contains(new Integer(j))) {
          pw.print(sep + "1");
        } else {
          pw.print(sep + "0");
        }
      }
    }
  }

  /**
   * prints the sparse feature matrix using {@link #printSparseFeatureMatrix()}
   * to {@link System#out System.out}.
   */
  public void printSparseFeatureMatrix() {
    printSparseFeatureMatrix(new PrintWriter(System.out, true));
  }

  /**
   * prints a sparse feature matrix representation of the Dataset.  Prints the actual
   * {@link Object#toString()} representations of features.
   */
  public void printSparseFeatureMatrix(PrintWriter pw) {
    String sep = "\t";
    for (int i = 0; i < size; i++) {
      pw.print(labelIndex.get(labels[i]));
      int[] datum = data[i];
      for (int j = 0; j < datum.length; j++) {
        pw.print(sep + featureIndex.get(datum[j]));
      }
      pw.println();
    }
  }

  public static void main(String[] args) {
    
    Dataset data = new Dataset();
    data.add(new BasicDatum(Arrays.asList(new String[]{"fever", "cough", "congestion"}), "cold"));
    data.add(new BasicDatum(Arrays.asList(new String[]{"fever", "cough", "nausea"}), "flu"));
    data.add(new BasicDatum(Arrays.asList(new String[]{"cough", "congestion"}), "cold"));
    data.summaryStatistics();
    
    data.applyFeatureCountThreshold(2);
     
    //Dataset data = Dataset.readSVMLightFormat(args[0]);
    //double[] scores = data.getInformationGains();
    //System.out.println(ArrayMath.mean(scores));
    //System.out.println(ArrayMath.variance(scores));
    LinearClassifierFactory factory = new LinearClassifierFactory();
    LinearClassifier classifier = (LinearClassifier)factory.trainClassifier(data);

    Datum d = new BasicDatum(Arrays.asList(new String[]{"cough", "fever"}));
    System.out.println(classifier.classOf(d));
    System.out.println(classifier.probabilityOf(d));

  }

  public void changeLabelIndex(Index newLabelIndex) {

    labels = trimToSize(labels);

    for (int i = 0; i < labels.length; i++) {
      labels[i] = newLabelIndex.indexOf(labelIndex.get(labels[i]));
    }
    labelIndex = newLabelIndex;
  }

  public void changeFeatureIndex(Index newFeatureIndex) {

    data = trimToSize(data);
    labels = trimToSize(labels);

    int[][] newData = new int[data.length][];
    for (int i = 0; i < data.length; i++) {
      int[] newD = new int[data[i].length];
      int k = 0;
      for (int j = 0; j < data[i].length; j++) {
        int newIndex = newFeatureIndex.indexOf(featureIndex.get(data[i][j]));
        if (newIndex >= 0) {
          newD[k++] = newIndex;
        }
      }
      newData[i] = new int[k];
      System.arraycopy(newD, 0, newData[i], 0, k);
    }
    data = newData;
    featureIndex = newFeatureIndex;
  }

  public void selectFeaturesBinaryInformationGain(int numFeatures) {

    double[] scores = getInformationGains();
    List<ScoredObject> scoredFeatures = new ArrayList<ScoredObject>();

    for (int i = 0; i < scores.length; i++) {
      scoredFeatures.add(new ScoredObject<Object>(featureIndex.get(i), scores[i]));
    }

    Collections.sort(scoredFeatures, ScoredComparator.DESCENDING_COMPARATOR);
    Index<Object> newFeatureIndex = new Index<Object>();
    for (int i = 0; i < scoredFeatures.size() && i < numFeatures; i++) {
      newFeatureIndex.add(scoredFeatures.get(i).object());
      //System.err.println(scoredFeatures.get(i));
    }

    for (int i = 0; i < size; i++) {
      int[] newData = new int[data[i].length];
      int curIndex = 0;
      for (int j = 0; j < data[i].length; j++) {
        int index;
        if ((index = newFeatureIndex.indexOf(featureIndex.get(data[i][j]))) != -1) {
          newData[curIndex++] = index;
        }
      }
      int[] newDataTrimmed = new int[curIndex];
      System.arraycopy(newData, 0, newDataTrimmed, 0, curIndex);
      data[i] = newDataTrimmed;
    }

    featureIndex = newFeatureIndex;
  }

  public double[] getInformationGains() {

    data = trimToSize(data);
    labels = trimToSize(labels);

    // counts the number of times word X is present
    ClassicCounter<Object> featureCounter = new ClassicCounter<Object>();

    // counts the number of time a document has label Y
    ClassicCounter<Object> labelCounter = new ClassicCounter<Object>();

    // counts the number of times the document has label Y given word X is present
    GeneralizedCounter condCounter = new GeneralizedCounter(2);

    for (int i = 0; i < labels.length; i++) {      
      labelCounter.incrementCount(labelIndex.get(labels[i]));

      // convert the document to binary feature representation
      boolean[] doc = new boolean[featureIndex.size()];
      //System.err.println(i);
      for (int j = 0; j < data[i].length; j++) {
        doc[data[i][j]] = true;
      }

      for (int j = 0; j < doc.length; j++) {
        if (doc[j]) {
          featureCounter.incrementCount(featureIndex.get(j));
          condCounter.incrementCount2D(featureIndex.get(j), labelIndex.get(labels[i]), 1.0);
        }
      }
    }

    double entropy = 0.0;
    for (int i = 0; i < labelIndex.size(); i++) {      
      double labelCount = labelCounter.getCount(labelIndex.get(i));
      double p = labelCount / (double)size();
      entropy -= p * (Math.log(p) / Math.log(2));
    }

    double[] ig = new double[featureIndex.size()];
    Arrays.fill(ig, entropy);

    for (int i = 0; i < featureIndex.size(); i++) {
      Object feature = featureIndex.get(i);
      
      double featureCount = featureCounter.getCount(feature);
      double notFeatureCount = size() - featureCount;

      double pFeature =  featureCount / (double)size();
      double pNotFeature = (1.0 - pFeature);

      if (featureCount == 0) { ig[i] = 0; continue; }
      if (notFeatureCount == 0) { ig[i] = 0; continue; }

      double sumFeature = 0.0;
      double sumNotFeature = 0.0;

      for (int j = 0; j < labelIndex.size(); j++) {
        Object label = labelIndex.get(j);

        double featureLabelCount = condCounter.getCount(feature, label);
        double notFeatureLabelCount = size() - featureLabelCount;

        // yes, these dont sum to 1.  that is correct.
        // one is the prob of the label, given that the
        // feature is present, and the other is the prob
        // of the label given that the feature is absent
        double p = featureLabelCount / featureCount;
        double pNot = notFeatureLabelCount / notFeatureCount;

        if (featureLabelCount != 0) { 
          sumFeature += p * (Math.log(p) / Math.log(2));
        }

        if (notFeatureLabelCount != 0) { 
          sumNotFeature += pNot * (Math.log(pNot) / Math.log(2));
        }
        //System.out.println(pNot+" "+(Math.log(pNot)/Math.log(2)));

      }

        //System.err.println(pFeature+" * "+sumFeature+" = +"+);
        //System.err.println("^ "+pNotFeature+" "+sumNotFeature);

      ig[i] = pFeature*sumFeature + pNotFeature*sumNotFeature;
    }
    return ig;
  }




  public String toString() {
    return "Dataset of size " + size;
  }

  public String toSummaryString() {
    StringWriter sw = new StringWriter();
    PrintWriter pw = new PrintWriter(sw);
    pw.println("Number of data points: " + size());
    pw.println("Number of active feature tokens: " + numFeatureTokens());
    pw.println("Number of active feature types:" + numFeatureTypes());
    return pw.toString();
  }

  /**
   * Need to sort the counter by feature keys and dump it
   *
   * @param pw
   * @param c
   * @param classNo
   */
  public static void printSVMLightFormat(PrintWriter pw, ClassicCounter<Integer> c, int classNo) {
    Integer[] features = c.keySet().toArray(new Integer[0]);
    Arrays.sort(features);
    StringBuilder sb = new StringBuilder();
    for (int f: features) {
      sb.append((f + 1) + ":" + c.getCount(f) + " ");
    }
    pw.println(classNo + " " + sb.toString());
  }

}
