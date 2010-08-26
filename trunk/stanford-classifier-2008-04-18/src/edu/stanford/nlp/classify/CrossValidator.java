package edu.stanford.nlp.classify;

import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;
import edu.stanford.nlp.util.Function;

import java.util.Iterator;

/**
 * This class is meant to simplify performing cross validation on
 * classifiers for hyper-parameters.  It has the ability to save
 * state for each fold (for instance, the weights for a MaxEnt
 * classifier, and the alphas for an SVM).
 *
 * @author Aria Haghighi
 * @author Jenny Finkel
 */

public class CrossValidator {
  private GeneralDataset originalTrainData;
  private int kfold;
  private int foldSize;
  private SavedState[] savedStates;

  public CrossValidator(GeneralDataset trainData) {
    this (trainData,5);
  }

  public CrossValidator(GeneralDataset trainData, int kfold) {
    originalTrainData = trainData;
    this.kfold = kfold;
    foldSize = (int)(((float) originalTrainData.size()) / kfold);
    savedStates = new SavedState[kfold];
    for (int i = 0; i < savedStates.length; i++) {
      savedStates[i] = new SavedState();
    }
  }

  /**
   * Returns and Iterator over train/test/saved states
   */
  private Iterator<Triple<GeneralDataset,GeneralDataset,SavedState>> iterator() { return new CrossValidationIterator(); }

  /**
   * This computes the average over all folds of the function we're trying to optimize.
   * The input triple contains, in order, the train set, the test set, and the saved state.  
   * You don't have to use the saved state if you don't want to.
   */
  public double computeAverage (Function<Triple<GeneralDataset,GeneralDataset,SavedState>,Double> function) 
  {
    double sum = 0;
    Iterator<Triple<GeneralDataset,GeneralDataset,SavedState>> foldIt = iterator();
    while (foldIt.hasNext()) {
      sum += function.apply(foldIt.next());
    }
    return sum / kfold;
  }

  class CrossValidationIterator implements Iterator<Triple<GeneralDataset,GeneralDataset,SavedState>>
  {
    int iter = 0;
    public boolean hasNext() { return iter < kfold; }

    public void remove()
    {
      throw new RuntimeException("CrossValidationIterator doesn't support remove()");
    }
  
    public Triple<GeneralDataset,GeneralDataset,SavedState> next()
    {
      if (iter == kfold) return null;
      int start = originalTrainData.size() * iter / kfold;
      int end = originalTrainData.size() * (iter + 1) / kfold;
      //System.err.println("##train data size: " +  originalTrainData.size() + " start " + start + " end " + end);
      Pair<GeneralDataset, GeneralDataset> split = originalTrainData.split(start, end);
      
      return new Triple<GeneralDataset,GeneralDataset,SavedState>(split.first(),split.second(),savedStates[iter++]);
    }
  }
  
  public static class SavedState {
    public Object state;
  }

  public static void main(String[] args) {
    Dataset d = Dataset.readSVMLightFormat(args[0]);
    Iterator<Triple<GeneralDataset,GeneralDataset,SavedState>> it = (new CrossValidator(d)).iterator();
    while (it.hasNext()) 
    { 
      Triple<GeneralDataset,GeneralDataset,SavedState> p = it.next(); 
      break;
    }
  }
}
