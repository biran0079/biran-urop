package edu.stanford.nlp.classify;

import edu.stanford.nlp.math.ArrayMath;
import java.io.Serializable;

/**
 * A Prior for functions.  Immutable.
 *
 * @author Galen Andrew
 */
public class LogPrior  implements Serializable {

  public enum LogPriorType { NULL, QUADRATIC, HUBER, QUARTIC, COSH, ADAPT }

  public static LogPriorType getType(String name) {
    if (name.equalsIgnoreCase("null")) { return LogPriorType.NULL; }
    else if (name.equalsIgnoreCase("quadratic")) { return LogPriorType.QUADRATIC; }
    else if (name.equalsIgnoreCase("huber")) { return LogPriorType.HUBER; }
    else if (name.equalsIgnoreCase("quartic")) { return LogPriorType.QUARTIC; }
    else if (name.equalsIgnoreCase("cosh")) { return LogPriorType.COSH; }
    else { throw new RuntimeException("Unknown LogPriorType: "+name); }
  }

  // these fields are just for the ADAPT prior -
  // is there a better way to do this?
  private double[] means = null;
  private LogPrior otherPrior = null;

  public static LogPrior getAdaptationPrior(double[] means, LogPrior otherPrior) {
    LogPrior lp = new LogPrior(LogPriorType.ADAPT);
    lp.means = means;
    lp.otherPrior = otherPrior;
    return lp;
  }
  
  public LogPriorType getType() {
    return type;
  }

  private final LogPriorType type;

  public LogPrior() {
    this(LogPriorType.QUADRATIC);
  }

  public LogPrior(int intPrior) {
    this(intPrior, 1.0, 0.1);
  }

  public LogPrior(LogPriorType type) {
    this(type, 1.0, 0.1);
  }

  // why isn't this functionality in enum?
  private static LogPriorType intToType(int intPrior) {
    LogPriorType[] values = LogPriorType.values();
    for (LogPriorType val : values) {
      if (val.ordinal() == intPrior) {
        return val;
      }
    }
    throw new IllegalArgumentException(intPrior + " is not a legal LogPrior.");
  }

  public LogPrior(int intPrior, double sigma, double epsilon) {
    this(intToType(intPrior), sigma, epsilon);
  }

  public LogPrior(LogPriorType type, double sigma, double epsilon) {
    this.type = type;
    setSigma(sigma);
    setEpsilon(epsilon);
  }

  private double sigma;
  private double sigmaSq;
  private double sigmaQu;
  private double epsilon;

  public double getSigma() {
    return sigma;
  }

  public double getEpsilon() {
    return epsilon;
  }

  public void setSigma(double sigma) {
    this.sigma = sigma;
    this.sigmaSq = sigma * sigma;
    this.sigmaQu = sigmaSq * sigmaSq;
  }

  public void setEpsilon(double epsilon) {
    this.epsilon = epsilon;
  }
  
  /**
   * Adjust the given grad array by adding the prior's gradient component
   * and return the value of the logPrior
   * @param x the input point
   * @param grad the gradient array
   * @return the value
   */
  public double compute(double[] x, double[] grad) {
    double val = 0.0;

    switch (type) {
      case NULL:
        return val;

      case QUADRATIC:
        for (int i = 0; i < x.length; i++) {
          val += x[i] * x[i] / 2.0 / sigmaSq;
          grad[i] += x[i] / sigmaSq;
        }
        return val;

      case HUBER:
        // P.J. Huber. 1973. Robust regression: Asymptotics, conjectures and
        // Monte Carlo. The Annals of Statistics 1: 799-821.
        // See also:
        // P. J. Huber. Robust Statistics. John Wiley & Sons, New York, 1981.
        for (int i = 0; i < x.length; i++) {
          if (x[i] < -epsilon) {
            val += (-x[i] - epsilon / 2.0) / sigmaSq;
            grad[i] += -1.0 / sigmaSq;
          } else if (x[i] < epsilon) {
            val += x[i] * x[i] / 2.0 / epsilon / sigmaSq;
            grad[i] += x[i] / epsilon / sigmaSq;
          } else {
            val += (x[i] - epsilon / 2.0) / sigmaSq;
            grad[i] += 1.0 / sigmaSq;
          }
        }
        return val;

      case QUARTIC:
        for (int i = 0; i < x.length; i++) {
          val += (x[i] * x[i]) * (x[i] * x[i]) / 2.0 / sigmaQu;
          grad[i] += x[i] / sigmaQu;
        }
        return val;

      case ADAPT:
        double[] newX = ArrayMath.pairwiseSubtract(x, means);
        val += otherPrior.compute(newX, grad);
        return val;

      case COSH:
        double norm = ArrayMath.norm_1(x) / sigmaSq;
        double d;
        if (norm > 30.0) {
          val = norm - Math.log(2);
          d = 1.0 / sigmaSq;
        } else {
          val = Math.log(Math.cosh(norm));
          d = (2 * (1 / (Math.exp(-2.0 * norm) + 1)) - 1.0) / sigmaSq;
        }
        for (int i=0; i < x.length; i++) {
          grad[i] += Math.signum(x[i]) * d;
        }
        return val;

      default:
        throw new RuntimeException("LogPrior.valueAt is undefined for prior of type " + this);
    }
  }


}
