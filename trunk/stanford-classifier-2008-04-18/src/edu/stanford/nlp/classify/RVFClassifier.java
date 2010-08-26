package edu.stanford.nlp.classify;

import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.ClassicCounter;
import java.io.Serializable;

/**
 * A simple interface for classifying and scoring data points with real
 * values features.  implemented by the linear classifier.
 *
 * @author Jenny Finkel
 */

public interface RVFClassifier extends Serializable {
  public Object classOf(RVFDatum example);

  public ClassicCounter scoresOf(RVFDatum example);
}
