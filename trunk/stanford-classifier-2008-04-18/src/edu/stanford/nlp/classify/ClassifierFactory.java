package edu.stanford.nlp.classify;

import java.io.Serializable;
import java.util.List;

/**
 * A simple interface for training a Classifier from a list of training
 * examples.
 *
 * @author Dan Klein
 */

public interface ClassifierFactory extends Serializable {
  public Classifier trainClassifier(List examples);
}
