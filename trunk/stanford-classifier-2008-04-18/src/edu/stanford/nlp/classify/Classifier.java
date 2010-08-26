package edu.stanford.nlp.classify;

import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.stats.ClassicCounter;

import java.io.Serializable;
import java.util.Collection;

/**
 * A simple interface for classifying and scoring data points, implemented
 * by most of the classifiers in this package.
 *
 * @author Dan Klein
 */

public interface Classifier extends Serializable {
  public Object classOf(Datum example);

  public ClassicCounter scoresOf(Datum example);

  public Collection labels();
}
