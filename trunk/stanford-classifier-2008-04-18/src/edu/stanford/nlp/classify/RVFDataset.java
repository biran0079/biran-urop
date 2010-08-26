package edu.stanford.nlp.classify;

import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.Pair;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.*;


/**
 * An interfacing class for {@link ClassifierFactory} that incrementally
 * builds a more memory-efficent representation of a {@link List} of
 * {@link RVFDatum} objects for the purposes of training a {@link Classifier}
 * with a {@link ClassifierFactory}.
 *
 * @author Jenny Finkel (jrfinkel@stanford.edu)
 * @author Rajat Raina (added methods to record data sources and ids)
 * @author Anna Rafferty (various refactoring with GeneralDataset/Dataset)
 */
public class RVFDataset extends GeneralDataset {

  private double[][] values;

  /*
   * Store source and id of each datum; optional, and not fully supported.
   */
  private ArrayList<Pair<String,String>> sourcesAndIds;

  public RVFDataset() {
    this(10);
  }

  public RVFDataset(int numDatums, Index featureIndex, Index labelIndex) {
    this(numDatums);
    this.labelIndex = labelIndex;
    this.featureIndex = featureIndex;
  }

  public RVFDataset(int numDatums) {
    initialize(numDatums);
  }

  /**
   * Constructor that fully specifies a Dataset.  Needed this for MulticlassDataset.
   */
  public RVFDataset(Index labelIndex, int[] labels, Index featureIndex, int[][] data, double[][] values) {
    this.labelIndex = labelIndex;
    this.labels = labels;
    this.featureIndex = featureIndex;
    this.data = data;
    this.values = values;
    this.size = labels.length;
  }

  public Pair<GeneralDataset, GeneralDataset> split(double percentDev) {
    int devSize = (int)(percentDev * size());
    int trainSize = size() - devSize;

    int[][] devData = new int[devSize][];
    double[][] devValues = new double[devSize][];
    int[] devLabels = new int[devSize];

    int[][] trainData = new int[trainSize][];
    double[][] trainValues = new double[trainSize][];
    int[] trainLabels = new int[trainSize];

    System.arraycopy(data, 0, devData, 0, devSize);
    System.arraycopy(values, 0, devValues, 0, devSize);
    System.arraycopy(labels, 0, devLabels, 0, devSize);

    System.arraycopy(data, devSize, trainData, 0, trainSize);
    System.arraycopy(values, devSize, trainValues, 0, trainSize);
    System.arraycopy(labels, devSize, trainLabels, 0, trainSize);


    RVFDataset dev = new RVFDataset(labelIndex, devLabels, featureIndex, devData, devValues);
    RVFDataset train = new RVFDataset(labelIndex, trainLabels, featureIndex, trainData, trainValues);

    return new Pair<GeneralDataset,GeneralDataset>(train, dev);

  }

  public Pair<GeneralDataset,GeneralDataset> split(int start, int end) {
    int devSize = end - start;
    int trainSize = size() - devSize;

    int[][] devData = new int[devSize][];
    double[][] devValues = new double[devSize][];
    int[] devLabels = new int[devSize];

    int[][] trainData = new int[trainSize][];
    double[][] trainValues = new double[trainSize][];
    int[] trainLabels = new int[trainSize];

    System.arraycopy(data, start, devData, 0, devSize);
    System.arraycopy(values, start, devValues, 0, devSize);
    System.arraycopy(labels, start, devLabels, 0, devSize);

    System.arraycopy(data, 0, trainData, 0, start);
    System.arraycopy(data, end, trainData, start, size()-end);
    System.arraycopy(values, 0, trainValues, 0, start);
    System.arraycopy(values, end, trainValues, start, size()-end);
    System.arraycopy(labels, 0, trainLabels, 0, start);
    System.arraycopy(labels, end, trainLabels, start, size()-end);

    GeneralDataset dev = new RVFDataset(labelIndex, devLabels, featureIndex, devData, devValues);
    GeneralDataset train = new RVFDataset(labelIndex, trainLabels, featureIndex, trainData, trainValues);

    return new Pair<GeneralDataset,GeneralDataset>(train, dev);

  }


  public void add(Datum d) {
    if (d instanceof RVFDatum) {
      addLabel(d.label());
      addFeatures(((RVFDatum)d).asFeaturesCounter());
      size++;
    } else {
      //addLabel(d.label());
      //addFeatures(d.asFeatures());
      //size++;
    }
  }

  public void add(Datum d, String src, String id) {
    if (d instanceof RVFDatum) {
      addLabel(d.label());
      addFeatures(((RVFDatum)d).asFeaturesCounter());
      addSourceAndId(src, id);
      size++;
    } else {
      //addLabel(d.label());
      //addFeatures(d.asFeatures());
      //size++;
    }
  }


  /**
   * @param index
   * @return the index-ed datum
   */
  public RVFDatum getRVFDatum(int index) {
    ClassicCounter<Object> c = new ClassicCounter<Object>();
    for (int i = 0; i < data[index].length; i++) {
      c.incrementCount(featureIndex.get(data[index][i]), values[index][i]);
    }
    return new RVFDatum(c, labelIndex.get(labels[index]));
  }

  public String getRVFDatumSource(int index) {
    return sourcesAndIds.get(index).first();
  }
  public String getRVFDatumId(int index) {
    return sourcesAndIds.get(index).second();
  }
  private void addSourceAndId(String src, String id) {
    sourcesAndIds.add(new Pair<String,String>(src, id));
  }

  private void addLabel(Object label) {
    if (labels.length == size) {
      int[] newLabels = new int[size * 2];
      System.arraycopy(labels, 0, newLabels, 0, size);
      labels = newLabels;
    }
    labelIndex.add(label);
    labels[size] = labelIndex.indexOf(label);
  }


  private void addFeatures(ClassicCounter features) {
    if (data.length == size) {
      int[][] newData = new int[size * 2][];
      double[][] newValues = new double[size * 2][];
      System.arraycopy(data, 0, newData, 0, size);
      System.arraycopy(values, 0, newValues, 0, size);
      data = newData;
      values = newValues;
    }
    int[] intFeatures = new int[features.size()];
    double[] featureValues = new double[features.size()];

    int j = 0;
    for (Iterator i = features.keySet().iterator(); i.hasNext();) {
      Object feature = i.next();
      featureIndex.add(feature);
      intFeatures[j] = featureIndex.indexOf(feature);
      featureValues[j] = features.getCount(feature);
      j++;
    }
    data[size] = intFeatures;
    values[size] = featureValues;
  }

  /**
   * Resets the Dataset so that it is empty and ready to collect data.
   */
  public void clear() {
    clear(10);
  }

  /**
   * Resets the Dataset so that it is empty and ready to collect data.
   */
  public void clear(int numDatums) {
    initialize(numDatums);
  }

  protected void initialize(int numDatums) {
    labelIndex = new Index<Object>();
    featureIndex = new Index<Object>();
    labels = new int[numDatums];
    data = new int[numDatums][];
    values = new double[numDatums][];
    sourcesAndIds = new ArrayList<Pair<String,String>>(numDatums);
    size = 0;
  }


  /**
   * Prints some summary statistics to stderr for the Dataset.
   */
  public void summaryStatistics() {
    System.err.println("numDatums: " + size);
    System.err.print("numLabels: " + labelIndex.size() + " [");
    Iterator iter = labelIndex.iterator();
    while (iter.hasNext()) {
      System.err.print(iter.next());
      if (iter.hasNext()) {
        System.err.print(", ");
      }
    }
    System.err.println("]");
    System.err.println("numFeatures (Phi(X) types): " + featureIndex.size());
    /*for(int i = 0; i < data.length; i++) {
      for(int j = 0; j < data[i].length; j++) {
      System.out.println(data[i][j]);
      }
      }*/
  }

//  private int[] trimToSize(int[] i, int size) {
//    int[] newI = new int[size];
//    System.arraycopy(i, 0, newI, 0, size);
//    return newI;
//  }
//
//  private int[][] trimToSize(int[][] i, int size) {
//    int[][] newI = new int[size][];
//    System.arraycopy(i, 0, newI, 0, size);
//    return newI;
//  }

  private double[][] trimToSize(double[][] i, int size) {
    double[][] newI = new double[size][];
    System.arraycopy(i, 0, newI, 0, size);
    return newI;
  }



  /**
   * prints the full feature matrix in tab-delimited form.  These can be BIG
   * matrices, so be careful! [Can also use printFullFeatureMatrixWithValues]
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
      pw.println();
    }
  }

  /**
   * Modification of printFullFeatureMatrix to correct bugs & print values (Rajat).
   * Prints the full feature matrix in tab-delimited form.  These can be BIG
   * matrices, so be careful!
   */
  public void printFullFeatureMatrixWithValues(PrintWriter pw) {
    String sep = "\t";
    for (int i = 0; i < featureIndex.size(); i++) {
      pw.print(sep + featureIndex.get(i));
    }
    pw.println();
    for (int i = 0; i < size; i++) { // changed labels.length to size
      pw.print(labelIndex.get(labels[i])); // changed i to labels[i]
      HashMap<Integer,Double> feats = new HashMap<Integer,Double>();
      for (int j = 0; j < data[i].length; j++) {
        int feature = data[i][j];
        double val = values[i][j];
        feats.put(new Integer(feature), new Double(val));
      }
      for (int j = 0; j < featureIndex.size(); j++) {
        if (feats.containsKey(new Integer(j))) {
          pw.print(sep + feats.get(new Integer(j)));
        } else {
          pw.print(sep + " ");
        }
      }
      pw.println();
    }
    pw.flush();
  }
  
  /**
   * Constructs a Dataset by reading in a file in SVM light format.
   * 
   */
  public static RVFDataset readSVMLightFormat(String filename) {
    return readSVMLightFormat(filename, new Index(), new Index());
  }

  /**
   * Constructs a Dataset by reading in a file in SVM light format.
   * The lines parameter is filled with the lines of the file for further processing
   * (if lines is null, it is assumed no line information is desired)
   */
  public static RVFDataset readSVMLightFormat(String filename, List<String> lines) {
    return readSVMLightFormat(filename, new Index(), new Index(), lines);
  }
  
  /**
   * Constructs a Dataset by reading in a file in SVM light format.
   * the created dataset has the same feature and label index as given
   */
  public static RVFDataset readSVMLightFormat(String filename, Index featureIndex, Index labelIndex) {
    return readSVMLightFormat(filename, featureIndex, labelIndex, null);
  }

  
  private static RVFDataset readSVMLightFormat(String filename, Index featureIndex, Index labelIndex, List<String> lines) {
    BufferedReader in = null;
    RVFDataset dataset;
    try {
      dataset = new RVFDataset(10, featureIndex, labelIndex);
      in = new BufferedReader(new FileReader(filename));

      while (in.ready()) {
        String line = in.readLine();
        if(lines != null)
          lines.add(line);
        dataset.add(svmLightLineToRVFDatum(line));
      }
      dataset.summaryStatistics();
      
    } catch (Exception e) {
      e.printStackTrace();
      throw new RuntimeException();
    }finally {
      if (in != null) {
        try {
          in.close();
        } catch (Exception ioe) {
        }
      }
    }
    return dataset;
  }

  public static RVFDatum svmLightLineToRVFDatum(String l) {
    String[] line = l.split("\\s+");
    ClassicCounter<String> features = new ClassicCounter<String>();
    for (int i = 1; i < line.length; i++) {
      String[] f = line[i].split(":");
      double val = Double.parseDouble(f[1]);
      features.incrementCount(f[0], val);
    }
    return new RVFDatum(features, line[0]);
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
    RVFDataset data = new RVFDataset();
    ClassicCounter<String> c1 = new ClassicCounter<String>();
    c1.incrementCount("fever", 3.5);
    c1.incrementCount("cough", 1.1);
    c1.incrementCount("congestion", 4.2);

    ClassicCounter<String> c2 = new ClassicCounter<String>();
    c2.incrementCount("fever", 1.5);
    c2.incrementCount("cough", 2.1);
    c2.incrementCount("nausea", 3.2);

    ClassicCounter<String> c3 = new ClassicCounter<String>();
    c3.incrementCount("cough", 2.5);
    c3.incrementCount("congestion", 3.2);

    data.add(new RVFDatum(c1, "cold"));
    data.add(new RVFDatum(c2, "flu"));
    data.add(new RVFDatum(c3, "cold"));
    data.summaryStatistics();

    LinearClassifierFactory factory = new LinearClassifierFactory();
    factory.useQuasiNewton();

    Classifier c = factory.trainClassifier(data);

    ClassicCounter<String> c4 = new ClassicCounter<String>();
    c4.incrementCount("cough", 2.3);
    c4.incrementCount("fever", 1.3);

    RVFDatum datum = new RVFDatum(c4);

    ((LinearClassifier) c).justificationOf(datum);
  }

  public double[][] getValuesArray() {
    values = trimToSize(values, size);
    return values;
  }


  public String toString() {
    return "Dataset of size " + size;
  }

  public String toSummaryString() {
    StringWriter sw = new StringWriter();
    PrintWriter pw = new PrintWriter(sw);
    pw.println("Number of data points: " + size());

    pw.print("Number of labels: " + labelIndex.size() + " [");
    Iterator iter = labelIndex.iterator();
    while (iter.hasNext()) {
      pw.print(iter.next());
      if (iter.hasNext()) {
        pw.print(", ");
      }
    }
    pw.println("]");
    pw.println("Number of features (Phi(X) types): " + featureIndex.size());
    pw.println("Number of active feature types: " + numFeatureTypes());
    pw.println("Number of active feature tokens: " + numFeatureTokens());

    return sw.toString();
  }

 
}
