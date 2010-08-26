package edu.stanford.nlp.classify;

import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.util.Index;

import java.lang.ref.Reference;
import java.util.Collection;

/**
 * Shared methods for training a {@link LinearClassifier}.
 * Inheriting classes need to implement the
 * <code>trainWeights</code> method.
 *
 * @author Dan Klein
 */

public abstract class AbstractLinearClassifierFactory {

  Index labelIndex = new Index();
  Index featureIndex = new Index();

  public AbstractLinearClassifierFactory() {
  }

  int numFeatures() {
    return featureIndex.size();
  }

  int numClasses() {
    return labelIndex.size();
  }

  protected abstract double[][] trainWeights(GeneralDataset dataset) ;

  /**
   * Takes a {@link Collection} of {@link Datum} objects and gives you back a
   * {@link Classifier} trained on it.
   *
   * @param examples {@link Collection} of {@link Datum} objects to train the
   *                 classifier on
   */
  public Classifier trainClassifier(Collection<Datum> examples) {
    Dataset dataset = new Dataset();
    dataset.addAll(examples);
    return trainClassifier(dataset);
  }

  /**
   * Takes a {@link Reference} to a {@link Collection} of {@link Datum}
   * objects and gives you back a {@link Classifier} trained on them
   *
   * @param ref {@link Reference} to a {@link Collection} of {@link
   *            Datum} objects to train the classifier on
   * @return A Classifier trained on a collection of Datum
   */
  public Classifier trainClassifier(Reference<Collection<Datum>> ref) {
    Collection<Datum> examples = ref.get();
    return trainClassifier(examples);
  }


  /**
   * trains a {@link Classifier} on a {@link Dataset}.
   *
   * @param data
   * @return a {@link Classifier} trained on the data.
   */
  public Classifier trainClassifier(GeneralDataset data) {
    labelIndex = data.labelIndex();
    featureIndex = data.featureIndex();
    double[][] weights = trainWeights(data);
    return new LinearClassifier(weights, featureIndex, labelIndex);
  }

}
