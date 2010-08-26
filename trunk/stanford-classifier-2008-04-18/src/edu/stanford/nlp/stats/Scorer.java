package edu.stanford.nlp.stats;

import edu.stanford.nlp.classify.GeneralDataset;
import edu.stanford.nlp.classify.ProbabilisticClassifier;

/**
 * @author Jenny Finkel
 */

public interface Scorer {

  public double score(ProbabilisticClassifier classifier, GeneralDataset data) ;

  public String getDescription(int numDigits);

} 
