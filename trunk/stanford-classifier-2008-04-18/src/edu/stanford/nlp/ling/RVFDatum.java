package edu.stanford.nlp.ling;

import edu.stanford.nlp.stats.ClassicCounter;

import java.util.Collection;
import java.util.Collections;

/**
 * Basic implementation of Datum interface that can be constructed with a
 * Collection of features and one more more labels. The features must be
 * specified
 * at construction, but the labels can be set and/or changed later.
 *
 * @author Jenny Finkel <a href="mailto:jrfinkel@stanford.edu">jrfinkel@stanford.edu</a>
 */
public class RVFDatum implements Datum {

  private static final long serialVersionUID = -255312811814660438L;

  /**
   * features for this Datum
   */
  private final ClassicCounter features;

  /**
   * labels for this Datum. Invariant: always non-null
   */
  private Object label = null;

  /**
   * Constructs a new RVFDatum with the given features and label.
   */
  public RVFDatum(ClassicCounter features, Object label) {
    this.features = features;
    setLabel(label);
  }

  /**
   * Constructs a new RVFDatum taking the data from a Datum
   *
   * @param m
   */
  public RVFDatum(Datum m) {
    this.features = new ClassicCounter();
    for (Object key : m.asFeatures()) {
      features.incrementCount(key, 1.0);
    }
    setLabel(m.label());
  }

  /**
   * Constructs a new RVFDatum with the given features and no labels.
   */
  public RVFDatum(ClassicCounter features) {
    this.features = features;
  }

  /**
   * Constructs a new RVFDatum with no features or labels.
   */
  public RVFDatum() {
    this((ClassicCounter) null);
  }

  /**
   * Returns the Counter of features and values
   */
  public ClassicCounter asFeaturesCounter() {
    return features;
  }

  /**
   * Returns the list of features without values
   */
  public Collection asFeatures() {
    return features.keySet();
  }


  /**
   * Removes all currently assigned Labels for this Datum then adds the
   * given Label.
   * Calling <tt>setLabel(null)</tt> effectively clears all labels.
   */
  public void setLabel(Object label) {
    this.label = label;
  }

  /**
   * Returns a String representation of this BasicDatum (lists features and labels).
   */
  public String toString() {
    return ("RVFDatum[features=" + asFeatures() + ",label=" + label() + "]");
  }

  public Object label() {
    return label;
  }

  public Collection labels() {
    return Collections.singletonList(label);
  }

  /**
   * Returns whether the given Datum contains the same features as this Datum.
   * Doesn't check the labels, should we change this?
   */
  public boolean equals(Object o) {
    if (!(o instanceof RVFDatum)) {
      return (false);
    }

    RVFDatum d = (RVFDatum) o;
    return (features.equals(d.asFeatures()));
  }

}

