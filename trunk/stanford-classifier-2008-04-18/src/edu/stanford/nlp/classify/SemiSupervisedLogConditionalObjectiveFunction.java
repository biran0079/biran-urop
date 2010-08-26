package edu.stanford.nlp.classify;

import edu.stanford.nlp.optimization.AbstractCachingDiffFunction;


/**
 * Maximizes the conditional likelihood with a given prior.
 *
 * @author Jenny Finkel
 */

public class SemiSupervisedLogConditionalObjectiveFunction extends AbstractCachingDiffFunction {

  LogConditionalObjectiveFunction objFunc;
  BiasedLogConditionalObjectiveFunction biasedObjFunc;  

  LogPrior prior;
  
  public void setPrior(LogPrior prior) {
    this.prior = prior;
  }
  
  public int domainDimension() {
    return objFunc.domainDimension();
  }

  protected void calculate(double[] x) {
    if (derivative == null) {
      derivative = new double[domainDimension()];
    }
    
    value = objFunc.valueAt(x) + biasedObjFunc.valueAt(x);
    double[] d1 = objFunc.derivativeAt(x);
    double[] d2 = biasedObjFunc.derivativeAt(x);

    for (int i = 0; i < domainDimension(); i++) {
      derivative[i] = d1[i] + d2[i];
    }
    value += prior.compute(x, derivative);
  }



  public SemiSupervisedLogConditionalObjectiveFunction(LogConditionalObjectiveFunction objFunc, BiasedLogConditionalObjectiveFunction biasedObjFunc, LogPrior prior) {
    this.objFunc = objFunc;
    this.biasedObjFunc = biasedObjFunc;
    this.prior = prior;
  }

}
