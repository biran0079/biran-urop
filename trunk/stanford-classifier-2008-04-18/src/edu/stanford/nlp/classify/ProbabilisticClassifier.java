package edu.stanford.nlp.classify;

import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.stats.ClassicCounter;

public interface ProbabilisticClassifier extends Classifier
{
  public ClassicCounter probabilityOf(Datum example);
  public ClassicCounter logProbabilityOf(Datum example);
}
