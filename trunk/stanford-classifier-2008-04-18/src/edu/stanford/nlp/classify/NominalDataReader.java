package edu.stanford.nlp.classify;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.StringTokenizer;

import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.util.FileLines;
import edu.stanford.nlp.util.Index;

/**
 * @author Kristina Toutanova
 *         Sep 14, 2004
 *         A class to read some UCI datasets into RVFDatum. Willl incrementally add formats
 */
public class NominalDataReader {
  HashMap indices = new HashMap(); // an Index for each feature so that its values are coded as integers

  /**
   * the class is the last column and it skips the next-to-last column because it is a unique id in the audiology data
   *
   * @param line
   * @return
   */
  RVFDatum readDatum(String line, String separator, HashMap indices) {
    StringTokenizer st = new StringTokenizer(line, separator);
    int fno = 0;
    ArrayList tokens = new ArrayList();
    while (st.hasMoreTokens()) {
      String token = st.nextToken();
      tokens.add(token);
    }
    Object[] arr = tokens.toArray();
    Set skip = new HashSet();
    skip.add(new Integer(arr.length - 2));
    return readDatum(arr, arr.length - 1, skip, indices);
  }

  RVFDatum readDatum(Object[] values, int classColumn, Set skip, HashMap indices) {
    ClassicCounter c = new ClassicCounter();
    RVFDatum d = new RVFDatum(c);
    int attrNo = 0;
    for (int index = 0; index < values.length; index++) {
      if (index == classColumn) {
        d.setLabel(values[index]);
        continue;
      }
      if (skip.contains(new Integer(index))) {
        continue;
      }
      Object featKey = new Integer(attrNo);
      Index ind = (Index) indices.get(featKey);
      if (ind == null) {
        ind = new Index();
        indices.put(featKey, ind);
      }
      if (!ind.isLocked()) {
        ind.add(values[index]);
      }
      int valInd = ind.indexOf(values[index]);
      if (valInd == -1) {
        valInd = 0;
        System.err.println("unknown attribute value " + values[index] + " of attribute " + attrNo);
      }
      c.incrementCount(featKey, valInd);
      attrNo++;

    }
    return d;
  }

  /**
   * Read the data as a list of RVFDatum objects. For the test set we must reuse the indices from the training set
   *
   * @param filename
   * @param indices
   * @return
   */
  ArrayList readData(String filename, HashMap indices) {
    try {
      
      String sep = ", ";
      ArrayList examples = new ArrayList();
      for(String line : new FileLines(filename)) {
        RVFDatum next = readDatum(line, sep, indices);
        examples.add(next);
      }
      return examples;
    } catch (Exception e) {
      e.printStackTrace();
    }
    return null;
  }


}
