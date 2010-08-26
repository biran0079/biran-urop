package edu.stanford.nlp.ling;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Basic implementation of Datum interface that can be constructed with a
 * Collection of features and one more more labels. The features must be
 * specified
 * at construction, but the labels can be set and/or changed later.
 *
 * @author Joseph Smarr (jsmarr@stanford.edu)
 */
public class BasicDatum implements Datum {
  /**
   * features for this Datum
   */
  private final Collection features;

  /**
   * labels for this Datum. Invariant: always non-null
   */
  private final List labels = new ArrayList();

  /**
   * Constructs a new BasicDatum with the given features and labels.
   */
  public BasicDatum(Collection features, Collection labels) {
    this(features);
    setLabels(labels);
  }

  /**
   * Constructs a new BasicDatum with the given features and label.
   */
  public BasicDatum(Collection features, Object label) {
    this(features);
    setLabel(label);
  }

  /**
   * Constructs a new BasicDatum with the given features and no labels.
   */
  public BasicDatum(Collection features) {
    this.features = features;
  }

  /**
   * Constructs a new BasicDatum with no features or labels.
   */
  public BasicDatum() {
    this(null);
  }

  /**
   * Returns the collection that this BasicDatum was constructed with.
   */
  public Collection asFeatures() {
    return (features);
  }

  /**
   * Returns the first label for this Datum, or null if none have been set.
   */
  public Object label() {
    return ((labels.size() > 0) ? (Object) labels.get(0) : null);
  }

  /**
   * Returns the complete List of labels for this Datum, which may be empty.
   */
  public Collection labels() {
    return labels;
  }

  /**
   * Removes all currently assigned Labels for this Datum then adds the
   * given Label.
   * Calling <tt>setLabel(null)</tt> effectively clears all labels.
   */
  public void setLabel(Object label) {
    labels.clear();
    addLabel(label);
  }

  /**
   * Removes all currently assigned labels for this Datum then adds all
   * of the given Labels.
   */
  public void setLabels(Collection labels) {
    this.labels.clear();
    if (labels != null) {
      this.labels.addAll(labels);
    }
  }

  /**
   * Adds the given Label to the List of labels for this Datum if it is not
   * null.
   */
  public void addLabel(Object label) {
    if (label != null) {
      labels.add(label);
    }
  }

  /**
   * Returns a String representation of this BasicDatum (lists features and labels).
   */
  public String toString() {
    return ("BasicDatum[features=" + asFeatures() + ",labels=" + labels() + "]");
  }


  /**
   * Returns whether the given Datum contains the same features as this Datum.
   * Doesn't check the labels, should we change this?
   */
  public boolean equals(Object o) {
    if (!(o instanceof Datum)) {
      return (false);
    }

    Datum d = (Datum) o;
    return (features.equals(d.asFeatures()));
  }

  private static final long serialVersionUID = -4857004070061779966L;

}

