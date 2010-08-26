package edu.stanford.nlp.classify;

import edu.stanford.nlp.math.ArrayMath;
import java.util.Arrays;


/**
 * Adapt the mean of the Gaussian Prior by shifting the mean to the previously trained weights
 * @author Pi-Chuan Chang
 */

public class AdaptedGaussianPriorObjectiveFunction extends LogConditionalObjectiveFunction {

  double[] weights;

  /**
   * Calculate the conditional likelihood.
   */
  protected void calculate(double[] x) {
    if (useSummedConditionalLikelihood) {
      calculateSCL(x);
    } else {
      calculateCL(x);
    }
  }


  /**
   * @param x
   */
  private void calculateSCL(double[] x) {
    throw new UnsupportedOperationException();
  }

  /**
   * @param x
   */
  private void calculateCL(double[] x) {
    value = 0.0;
    if (derivativeNumerator == null) {
      derivativeNumerator = new double[x.length];
      for (int d = 0; d < data.length; d++) {
        int[] features = data[d];
        for (int f = 0; f < features.length; f++) {
          int i = indexOf(features[f], labels[d]);
          if (dataweights == null) {
            derivativeNumerator[i] -= 1;
          } else {
            derivativeNumerator[i] -= dataweights[d];
          }
        }
      }
    }
    copy(derivative, derivativeNumerator);

    double[] sums = new double[numClasses];
    double[] probs = new double[numClasses];

    for (int d = 0; d < data.length; d++) {
      int[] features = data[d];
      // activation
      Arrays.fill(sums, 0.0);

      for (int c = 0; c < numClasses; c++) {
        for (int f = 0; f < features.length; f++) {
          int i = indexOf(features[f], c);
          sums[c] += x[i];
        }
      }
      double total = ArrayMath.logSum(sums);
      for (int c = 0; c < numClasses; c++) {
        probs[c] = Math.exp(sums[c] - total);
        if (dataweights != null) {
          probs[c] *= dataweights[d];
        }
        for (int f = 0; f < features.length; f++) {
          int i = indexOf(features[f], c);
          derivative[i] += probs[c];
        }
      }

      double dV = sums[labels[d]] - total;
      if (dataweights != null) {
        dV *= dataweights[d];
      }
      value -= dV;
    }
    //System.err.println("x length="+x.length);
    //System.err.println("weights length="+weights.length);
    double[] newX = ArrayMath.pairwiseSubtract(x, weights);
    value += prior.compute(newX, derivative);
  }

  /**
   * @param x
   */
  protected void rvfcalculate(double[] x) {
    throw new UnsupportedOperationException();
  }

  public AdaptedGaussianPriorObjectiveFunction(GeneralDataset dataset, LogPrior prior, double weights[][]) {
    super(dataset, prior);
    this.weights = to1D(weights);
  }

  public double[] to1D(double[][] x2) {
    double[] x = new double[numFeatures*numClasses];
    for (int i = 0; i < numFeatures; i++) {
      for (int j = 0; j < numClasses; j++) {
        x[indexOf(i, j)] = x2[i][j];
      }
    }
    return x;
  }
}
