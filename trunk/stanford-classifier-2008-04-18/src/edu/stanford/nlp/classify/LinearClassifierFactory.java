// Stanford Classifier - a multiclass maxent classifier
// LinearClassifierFactory
// Copyright (c) 2003-2007 The Board of Trustees of
// The Leland Stanford Junior University. All Rights Reserved.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//
// For more information, bug reports, fixes, contact:
//    Christopher Manning
//    Dept of Computer Science, Gates 1A
//    Stanford CA 94305-9010
//    USA
//    java-nlp-support@lists.stanford.edu
//    http://www-nlp.stanford.edu/software/classifier.shtml

package edu.stanford.nlp.classify;

import java.io.*;
import edu.stanford.nlp.optimization.*;
import edu.stanford.nlp.util.Function;
import edu.stanford.nlp.stats.MultiClassAccuracyStats;
import edu.stanford.nlp.stats.Scorer;
import edu.stanford.nlp.util.ArrayUtils;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Timing;
import edu.stanford.nlp.util.Triple;
import edu.stanford.nlp.sequences.SeqClassifierFlags;

import java.util.Arrays;

/**
 * Builds various types of linear classifiers, with functionality for
 * setting objective function, optimization method, and other parameters.
 * Classifiers can be defined with passed constructor arguments or using setter methods.
 * Defaults to Quasi-newton optimization of a <code>LogConditionalObjectiveFunction</code>
 * (Merges old classes: CGLinearClassifierFactory, QNLinearClassifierFactory, and MaxEntClassifierFactory).
 *
 * @author Jenny Finkel
 * @author Chris Cox (merged factories, 8/11/04)
 * @author Dan Klein (CGLinearClassifierFactory, MaxEntClassifierFactory)
 * @author Galen Andrew (tuneSigma), Marie-Catherine de Marneffe (CV in tuneSigma)
 */

public class LinearClassifierFactory extends AbstractLinearClassifierFactory {

  private double TOL;
  //public double sigma;
  private int mem = 15;
  private boolean verbose = false;
  //private int prior;
  //private double epsilon = 0.0;
  private LogPrior logPrior;
  private Minimizer minimizer;
  private boolean useSum = false;
  private boolean tuneSigmaHeldOut = false;
  private boolean tuneSigmaCV = false;
  private boolean resetWeight = true;
  private int folds;
  private double min = 0.1;
  private double max = 10.0;
  private boolean retrainFromScratchAfterSigmaTuning = false;


  /**
   * Adapt classifier (adjust the mean of Gaussian prior)
   * under construction -pichuan
   * @param origWeights the original weights trained from the training data
   * @param adaptDataset the Dataset used to adapt the trained weights
   * @return adapted weights
   */
  public double[][] adaptWeights(double[][] origWeights, GeneralDataset adaptDataset) {
    System.err.println("adaptWeights in LinearClassifierFactory. increase weight dim only");
    double[][] newWeights = new double[adaptDataset.featureIndex.size()][adaptDataset.labelIndex.size()];

    System.arraycopy(origWeights,0,newWeights,0,origWeights.length);

    AdaptedGaussianPriorObjectiveFunction objective = new AdaptedGaussianPriorObjectiveFunction(adaptDataset, logPrior,newWeights);
    
    double[] initial = objective.initial();

    double[] weights = minimizer.minimize(objective, TOL, initial);
    return objective.to2D(weights);

    //Question: maybe the adaptWeights can be done just in LinearClassifier ?? (pichuan)
  }

  public double[][] trainWeights(GeneralDataset dataset) {
    return trainWeights(dataset, null);
  }

  public double[][] trainWeights(GeneralDataset dataset, double[] initial) {
    return trainWeights(dataset, initial, false);
  }
  
  public double[][] trainWeights(GeneralDataset dataset, double[] initial, boolean bypassTuneSigma) {
    double[] interimWeights = null;
    if(! bypassTuneSigma) {
      if (tuneSigmaHeldOut) {
        interimWeights = heldOutSetSigma(dataset); // the optimum interim weights from held-out training data have already been found.
      } else if (tuneSigmaCV) {
        crossValidateSetSigma(dataset,folds); // TODO: assign optimum interim weights as part of this process.
      }
    }
    LogConditionalObjectiveFunction objective = new LogConditionalObjectiveFunction(dataset, logPrior);
    if(initial == null && interimWeights != null && ! retrainFromScratchAfterSigmaTuning) {
      //System.err.println("## taking advantage of interim weights as starting point.");
      initial = interimWeights; 
    }
    if (initial == null) {
      initial = objective.initial();
    }

    double[] weights = minimizer.minimize(objective, TOL, initial);
    return objective.to2D(weights);
  }

  /**
   * IMPORTANT: dataset and biasedDataset must have same featureIndex, labelIndex
   */
  public Classifier trainClassifierSemiSup(GeneralDataset data, GeneralDataset biasedData, double[][] confusionMatrix, double[] initial) {
    double[][] weights =  trainWeightsSemiSup(data, biasedData, confusionMatrix, initial);
    LinearClassifier classifier = new LinearClassifier(weights, data.featureIndex(), data.labelIndex());
    return classifier;
  }
  
  public double[][] trainWeightsSemiSup(GeneralDataset data, GeneralDataset biasedData, double[][] confusionMatrix, double[] initial) {
    LogConditionalObjectiveFunction objective = new LogConditionalObjectiveFunction(data, new LogPrior(LogPrior.LogPriorType.NULL));
    BiasedLogConditionalObjectiveFunction biasedObjective = new BiasedLogConditionalObjectiveFunction(biasedData, confusionMatrix, new LogPrior(LogPrior.LogPriorType.NULL));
    SemiSupervisedLogConditionalObjectiveFunction semiSupObjective = new SemiSupervisedLogConditionalObjectiveFunction(objective, biasedObjective, logPrior);
    if (initial == null) {
      initial = objective.initial();
    }
    double[] weights = minimizer.minimize(semiSupObjective, TOL, initial);
    return objective.to2D(weights);
  }

  
  /**
   * Train a classifier with a sigma tuned on a validation set.
   *
   * @param train
   * @param validation
   * @return The constructed classifier
   */
  public Classifier trainClassifierV(GeneralDataset train, GeneralDataset validation, double min, double max, boolean accuracy) {
    labelIndex = train.labelIndex();
    featureIndex = train.featureIndex();
    this.min = min;
    this.max = max;
    heldOutSetSigma(train, validation);
    double[][] weights = trainWeights(train);    
    return new LinearClassifier(weights, train.featureIndex(), train.labelIndex());
  }

  /**
   * Train a classifier with a sigma tuned on a validation set.
   * In this case we are fitting on the last 30% of the training data.
   *
   * @param train The data to train (and validate) on.
   * @return The constructed classifier
   */
  public Classifier trainClassifierV(GeneralDataset train, double min, double max, boolean accuracy) {
    labelIndex = train.labelIndex();
    featureIndex = train.featureIndex();
    tuneSigmaHeldOut = true;
    this.min = min;
    this.max = max;
    heldOutSetSigma(train);
    double[][] weights = trainWeights(train);    
    return new LinearClassifier(weights, train.featureIndex(), train.labelIndex());
  }


  public LinearClassifierFactory() {
    this(new QNMinimizer(15));
  };

  public LinearClassifierFactory(Minimizer min) {
    this(min, false);
  };

  public LinearClassifierFactory(boolean useSum) {
    this(new QNMinimizer(15), useSum);
  };

  public LinearClassifierFactory(double tol) {
    this(new QNMinimizer(15), tol, false);
  };
  public LinearClassifierFactory(Minimizer min, boolean useSum) {
    this(min, 1e-4, useSum);
  };
  public LinearClassifierFactory(Minimizer min, double tol, boolean useSum) {
    this(min, tol, useSum, 1.0);
  };
  public LinearClassifierFactory(double tol, boolean useSum, double sigma) {
    this(new QNMinimizer(15), tol, useSum, sigma);
  };
  public LinearClassifierFactory(Minimizer min, double tol, boolean useSum, double sigma) {
    this(min, tol, useSum, LogPrior.LogPriorType.QUADRATIC.ordinal(), sigma);
  };
  public LinearClassifierFactory(Minimizer min, double tol, boolean useSum, int prior, double sigma) {
    this(min, tol, useSum, prior, sigma, 0.0);
  };
  public LinearClassifierFactory(double tol, boolean useSum, int prior, double sigma, double epsilon) {    
    this(new QNMinimizer(15), tol, useSum, new LogPrior(prior, sigma, epsilon));
  }

  public LinearClassifierFactory(double tol, boolean useSum, int prior, double sigma, double epsilon, int mem) {
    this(new QNMinimizer(mem), tol, useSum, new LogPrior(prior, sigma, epsilon));
  };

  /**
   * Create a factory that builds linear classifiers from training data.
   *
   * @param min     The method to be used for optimization (minimization) (default: {@link QNMinimizer})
   * @param tol     The convergence threshold for the minimization (default: 1e-4)
   * @param useSum  Asks to the optimizer to minimize the sum of the
   *                likelihoods of individual data items rather than their product (default: false)
   * @param prior   What kind of prior to use, as an enum constant from class
   *                LogPrior 
   * @param sigma   The strength of the prior (smaller is stronger for most
   *                standard priors) (default: 1.0)
   * @param epsilon A second parameter to the prior (currently only used
   *                by the Huber prior)
   */
  public LinearClassifierFactory(Minimizer min, double tol, boolean useSum, int prior, double sigma, double epsilon) {
    this(min, tol, useSum, new LogPrior(prior, sigma, epsilon));
  }

  public LinearClassifierFactory(Minimizer min, double tol, boolean useSum, LogPrior logPrior) {
    this.minimizer = min;
    this.TOL = tol;
    this.useSum = useSum;
    this.logPrior = logPrior;
  };

  /**
   * Set the tolerance.  1e-4 is the default.
   */
  public void setTol(double tol) {
    this.TOL = tol;
  }

  /**
   * Set the prior.
   *
   * @param logPrior One of the priors defined in 
   *              <code>LogConditionalObjectiveFunction</code>.
   *              <code>LogPrior.QUADRATIC</code> is the default.
   */
  public void setPrior(LogPrior logPrior) {
    this.logPrior = logPrior;
  }

  /**
   * Set the verbose flag for {@link CGMinimizer}.
   * Only used with conjugate-gradient minimization. 
   * <code>false</code> is the default.
   */

  public void setVerbose(boolean verbose) {
    this.verbose = verbose;
  }

  /**
   * Sets the minimizer.  {@link QNMinimizer} is the default.
   */
  public void setMinimizer(Minimizer min) {
    this.minimizer = min;
  }

  /**
   * Sets the epsilon value for {@link LogConditionalObjectiveFunction}.
   */
  public void setEpsilon(double eps) {
    logPrior.setEpsilon(eps);
  }

  public void setSigma(double sigma) {
    logPrior.setSigma(sigma);
  }

  public double getSigma() {
    return logPrior.getSigma();
  }

  /**
   * Sets the minimizer to QuasiNewton. {@link QNMinimizer} is the default.
   */
  public void useQuasiNewton() {
    this.minimizer = new QNMinimizer(mem);
  }
  
  public void useQuasiNewton(boolean useRobust) {
    this.minimizer = new QNMinimizer(mem,useRobust);
  }

  public void useStochasticQN(double initialSMDGain, int stochasticBatchSize){
    this.minimizer = new SQNMinimizer(mem,initialSMDGain,stochasticBatchSize,false);
  }
  
  public void useStochasticMetaDescent(){
    useStochasticMetaDescent(0.1,15,StochasticCalculateMethods.ExternalFiniteDifference,20);
  }

  public void useStochasticMetaDescent(double initialSMDGain, int stochasticBatchSize,StochasticCalculateMethods stochasticMethod,int passes) {
    this.minimizer = new SMDMinimizer(initialSMDGain, stochasticBatchSize,stochasticMethod,passes);
  }

  public void useStochasticGradientDescent(){
    useStochasticGradientDescent(0.1,15);
  }

  public void useStochasticGradientDescent(double gainSGD, int stochasticBatchSize){
    this.minimizer = new SGDMinimizer(gainSGD,stochasticBatchSize);
  }

  public void useStochasticGradientDescentToQuasiNewton(SeqClassifierFlags p){
    this.minimizer = new SGDToQNMinimizer(p);
  }

  public void useHybridMinimizer(){
    useHybridMinimizer(0.1,15,StochasticCalculateMethods.ExternalFiniteDifference ,0);
  }

  public void useHybridMinimizer(double initialSMDGain, int stochasticBatchSize,StochasticCalculateMethods stochasticMethod,int cutoffIteration){
    Minimizer firstMinimizer = new SMDMinimizer(initialSMDGain, stochasticBatchSize,stochasticMethod,cutoffIteration);
    Minimizer secondMinimizer = new QNMinimizer(mem);
    this.minimizer = new HybridMinimizer(firstMinimizer,secondMinimizer,cutoffIteration);
  }

  /**
   * Set the mem value for {@link QNMinimizer}.
   * Only used with quasi-newton minimization.  15 is the default.
   *
   * @param mem Number of previous function/derivative evaluations to store
   *            to estimate second derivative.  Storing more previous evaluations
   *            improves training convergence speed.  This number can be very
   *            small, if memory conservation is the priority.  For large
   *            optimization systems (of 100,000-1,000,000 dimensions), setting this
   *            to 15 produces quite good results, but setting it to 50 can
   *            decrease the iteration count by about 20% over a value of 15.
   */
  public void setMem(int mem) {
    this.mem = mem;
  }

  /**
   * Sets the minimizer to {@link CGMinimizer}, with the passed <code>verbose</code> flag.
   */
  public void useConjugateGradientAscent(boolean verbose) {
    this.verbose = verbose;
    useConjugateGradientAscent();
  }

  /**
   * Sets the minimizer to {@link CGMinimizer}.
   */
  public void useConjugateGradientAscent() {
    this.minimizer = new CGMinimizer(!this.verbose);
  }

  /**
   * SetUseSum sets the <code>useSum</code> flag: when turned on,
   * the Summed Conditional Objective Function is used.  Otherwise, the
   * LogConditionalObjectiveFunction is used.  The default is false.
   */
  public void setUseSum(boolean useSum) {
    this.useSum = useSum;
  }

  /**
   * setTuneSigmaHeldOut sets the <code>tuneSigmaHeldOut</code> flag: when turned on,
   * the sigma is tuned by means of held-out (70%-30%). Otherwise no tuning on sigma is done.
   * The default is false.
   */
  public void setTuneSigmaHeldOut() {
    tuneSigmaHeldOut = true;
    tuneSigmaCV = false;
  }

  /**
   * setTuneSigmaCV sets the <code>tuneSigmaCV</code> flag: when turned on,
   * the sigma is tuned by cross-validation. The number of folds is the parameter.
   * If there is less data than the number of folds, leave-one-out is used.
   * The default is false.
   */
  public void setTuneSigmaCV(int folds) {
    tuneSigmaCV = true;
    tuneSigmaHeldOut = false;
    this.folds = folds;
  }

  /**
   * resetWeight sets the <code>restWeight</code> flag. This flag makes sense only if sigma is tuned:
   * when turned on, the weights outputed by the tuneSigma method will be reset to zero when training the
   * classifier.
   * The default is false.
   */
  public void resetWeight() {
    resetWeight = true;
  }

  static protected double[] sigmasToTry = {0.5,1.0,2.0,4.0,10.0, 20.0, 100.0};

  /**
   * Calls the method {@link #crossValidateSetSigma(GeneralDataset, int)} with 5-fold cross-validation. 
   * @param dataset the data set to optimize sigma on.
   */
  public void crossValidateSetSigma(GeneralDataset dataset) {
    crossValidateSetSigma(dataset, 5);
  }

  /**
   * callls the method {@link #crossValidateSetSigma(GeneralDataset, int, Scorer, LineSearcher)} with 
   * multi-class log-likelihood scoring (see {@link MultiClassAccuracyStats}) and golden-section line search
   * (see {@link GoldenSectionLineSearch}). 
   * @param dataset the data set to optimize sigma on.
   * @param kfold
   */
  public void crossValidateSetSigma(GeneralDataset dataset,int kfold) {
    System.err.println("##you are here.");
    crossValidateSetSigma(dataset, kfold, new MultiClassAccuracyStats(MultiClassAccuracyStats.USE_LOGLIKELIHOOD), new GoldenSectionLineSearch(true, 1e-2, min, max));
  }

  public void crossValidateSetSigma(GeneralDataset dataset,int kfold, final Scorer scorer) {
    crossValidateSetSigma(dataset, kfold, scorer, new GoldenSectionLineSearch(true, 1e-2, min, max));
  }
  public void crossValidateSetSigma(GeneralDataset dataset,int kfold, LineSearcher minimizer) {
    crossValidateSetSigma(dataset, kfold, new MultiClassAccuracyStats(MultiClassAccuracyStats.USE_LOGLIKELIHOOD), minimizer);
  }
  /**
   * Sets the sigma parameter to a value that optimizes the cross-validation score given by <code>scorer</code>.  Search for an optimal value
   * is carried out by <code>minimizer</code> 
   * @param dataset the data set to optimize sigma on.
   * @param kfold
   */  
  public void crossValidateSetSigma(GeneralDataset dataset,int kfold, final Scorer scorer, LineSearcher minimizer) {
    System.err.println("##in Cross Validate, folds = " + kfold);
    System.err.println("##Scorer is " + scorer);

    featureIndex = dataset.featureIndex;
    labelIndex = dataset.labelIndex;

    final CrossValidator crossValidator = new CrossValidator(dataset,kfold);
    final Function<Triple<GeneralDataset,GeneralDataset,CrossValidator.SavedState>,Double> score = 
      new Function<Triple<GeneralDataset,GeneralDataset,CrossValidator.SavedState>,Double> () 
      {
        public Double apply (Triple<GeneralDataset,GeneralDataset,CrossValidator.SavedState> fold) {
          GeneralDataset trainSet = fold.first();
          GeneralDataset devSet   = fold.second();

          double[] weights = (double[])fold.third().state;
          double[][] weights2D;

          weights2D = trainWeights(trainSet, weights,true); // must of course bypass sigma tuning here.

          fold.third().state = ArrayUtils.flatten(weights2D);

          LinearClassifier classifier = new LinearClassifier(weights2D, trainSet.featureIndex, trainSet.labelIndex);
          
          double score = scorer.score(classifier, devSet);
          //System.out.println("score: "+score);
          System.out.print(".");
          return score;
        }
      };
    
    Function<Double,Double> negativeScorer = 
      new Function<Double,Double> ()
      {
        public Double apply(Double sigmaToTry) {          
          //sigma = sigmaToTry;
          setSigma(sigmaToTry);
          Double averageScore = crossValidator.computeAverage(score);
          System.err.print("##sigma = "+getSigma()+" ");
          System.err.println("-> average Score: "+averageScore);
          return -averageScore;
        }
      };      
    
    double bestSigma = minimizer.minimize(negativeScorer);
    System.err.println("##best sigma: " + bestSigma);
    setSigma(bestSigma);
  }
  
  /**
   * Set the {@link LineSearcher} to be used in {@link #heldOutSetSigma(GeneralDataset, GeneralDataset)}.
   */
  public void setHeldOutSearcher(LineSearcher heldOutSearcher) {
    this.heldOutSearcher = heldOutSearcher;
  }

  private LineSearcher heldOutSearcher = null;
  public double[] heldOutSetSigma(GeneralDataset train) {
    Pair<GeneralDataset, GeneralDataset> data = train.split(0.3);
    return heldOutSetSigma(data.first(), data.second());
  }

  public double[] heldOutSetSigma(GeneralDataset train, Scorer scorer) {
    Pair<GeneralDataset, GeneralDataset> data = train.split(0.3);
    return heldOutSetSigma(data.first(), data.second(), scorer);
  }

  public double[] heldOutSetSigma(GeneralDataset train, GeneralDataset dev) {
    return heldOutSetSigma(train, dev, new MultiClassAccuracyStats(MultiClassAccuracyStats.USE_LOGLIKELIHOOD), heldOutSearcher == null ? new GoldenSectionLineSearch(true, 1e-2, min, max) : heldOutSearcher);
  }

  public double[] heldOutSetSigma(GeneralDataset train, GeneralDataset dev, final Scorer scorer) {
    return heldOutSetSigma(train, dev, scorer, new GoldenSectionLineSearch(true, 1e-2, min, max));
  }
  public double[]  heldOutSetSigma(GeneralDataset train, GeneralDataset dev, LineSearcher minimizer) {
    return heldOutSetSigma(train, dev, new MultiClassAccuracyStats(MultiClassAccuracyStats.USE_LOGLIKELIHOOD), minimizer);
  }
  
  /**
   * Sets the sigma parameter to a value that optimizes the held-out score given by <code>scorer</code>.  Search for an optimal value
   * is carried out by <code>minimizer</code> 
   * dataset the data set to optimize sigma on.
   * kfold
   * @return an interim set of optimal weights: the weights 
   */  
  public double[] heldOutSetSigma(final GeneralDataset trainSet, final GeneralDataset devSet, final Scorer scorer, LineSearcher minimizer) {

    featureIndex = trainSet.featureIndex;
    labelIndex = trainSet.labelIndex;
    double[] resultWeights = null;
    Timing timer = new Timing();
    
    NegativeScorer negativeScorer = new NegativeScorer(trainSet,devSet,scorer,timer);
    
    timer.start();
    double bestSigma = minimizer.minimize(negativeScorer);
    System.err.println("##best sigma: " + bestSigma);
    setSigma(bestSigma);
    
    return ArrayUtils.flatten(trainWeights(trainSet,negativeScorer.weights,true)); // make sure it's actually the interim weights from best sigma
  }
  
  class NegativeScorer implements Function<Double, Double> {
    public double[] weights = null;
    GeneralDataset trainSet;
    GeneralDataset devSet;
    Scorer scorer;
    Timing timer;
    
    public NegativeScorer(GeneralDataset trainSet, GeneralDataset devSet, Scorer scorer,Timing timer) {
      super();
      this.trainSet = trainSet;
      this.devSet = devSet;
      this.scorer = scorer;
      this.timer = timer;
    }

    public Double apply(Double sigmaToTry) {         
      double[][] weights2D;
      setSigma(sigmaToTry);
      
      weights2D = trainWeights(trainSet, weights,true); //bypass.
      
      weights = ArrayUtils.flatten(weights2D);
                
      LinearClassifier classifier = new LinearClassifier(weights2D, trainSet.featureIndex, trainSet.labelIndex);
      
      double score = scorer.score(classifier, devSet);
      //System.out.println("score: "+score);
      //System.out.print(".");
      System.err.print("##sigma = "+getSigma()+" ");
      System.err.println("-> average Score: "+ score);
      System.err.println("##time elapsed: " + timer.stop() + " milliseconds.");
      timer.restart();
      return -score;
    }    
  }

  /** If set to true, then when training a classifier, after an optimal sigma is chosen a model is relearned from
   * scratch. If set to false (the default), then the model is updated from wherever it wound up in the sigma-tuning process.
   * The latter is likely to be faster, but it's not clear which model will wind up better.  */
  public void setRetrainFromScratchAfterSigmaTuning( boolean retrainFromScratchAfterSigmaTuning) {
    this.retrainFromScratchAfterSigmaTuning = retrainFromScratchAfterSigmaTuning;
  }

  public Classifier trainClassifier(GeneralDataset dataset, float[] dataWeights, LogPrior prior) {
    LogConditionalObjectiveFunction objective = new LogConditionalObjectiveFunction(dataset, dataWeights, logPrior);
    
    double[] initial = objective.initial();
    double[] weights = minimizer.minimize(objective, TOL, initial);

    LinearClassifier classifier = new LinearClassifier(objective.to2D(weights), dataset.featureIndex(), dataset.labelIndex());
    return
      classifier;
  }

  
  public Classifier trainClassifier(GeneralDataset dataset) {
    return trainClassifier(dataset, null);
  }

  public Classifier trainClassifier(GeneralDataset dataset, double[] initial) {
    double[][] weights =  trainWeights(dataset, initial, false);
    LinearClassifier classifier = new LinearClassifier(weights, dataset.featureIndex(), dataset.labelIndex());
    return classifier;
  }
  
  /**
   * Given the path to a file representing the text based serialization of a 
   * Linear Classifier, reconstitutes and returns that LinearClassifier.
   * 
   * TODO: Leverage Index
   * @param file
   * @return
   */
  public Classifier loadFromFilename(String file) {
    try {
      File tgtFile = new File(file);     
      BufferedReader in = new BufferedReader(new FileReader(tgtFile));
      
      // Format: read indicies first, weights, then thresholds
      Index labelIndex = Index.loadFromReader(in);
      Index featureIndex = Index.loadFromReader(in);
      double weights[][] = new double[featureIndex.size()][labelIndex.size()];
      String line = in.readLine();
      int currLine = 1;
      while (line != null && line.length()>0) {
        String[] tuples = line.split(LinearClassifier.TEXT_SERIALIZATION_DELIMITER);
        if (tuples.length != 3) { 
            throw new Exception("Error: incorrect number of tokens in weight specifier, line="
            +currLine+" in file "+tgtFile.getAbsolutePath()); 
        }
        currLine++;
        int feature = Integer.valueOf(tuples[0]);
        int label = Integer.valueOf(tuples[1]);
        double value = Double.valueOf(tuples[2]);
        weights[feature][label] = value;
        line = in.readLine();
      }
      
      // First line in thresholds is the number of thresholds
      int numThresholds = Integer.valueOf(in.readLine());
      double[] thresholds = new double[numThresholds];
      int curr = 0;
      while ((line = in.readLine()) != null) {
        double tval = Double.valueOf(line.trim());
        thresholds[curr++] = tval;
      }
      LinearClassifier classifier = new LinearClassifier(weights, featureIndex, labelIndex);
      return classifier;
    } catch (Exception e) {
      System.err.println("Error in LinearClassifierFactory, loading from file="+file);
      e.printStackTrace();
      return null;
    }
  }

}
