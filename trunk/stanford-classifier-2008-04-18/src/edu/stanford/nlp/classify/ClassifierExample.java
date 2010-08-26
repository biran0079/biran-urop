package edu.stanford.nlp.classify;

import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;

import java.util.ArrayList;
import java.util.List;

/**
 * Sample code that illustrates the training and use of a linear classifier.
 * @author Dan Klein
 */
public class ClassifierExample {

  protected static final String GREEN = "green";
  protected static final String RED = "red";
  protected static final String WORKING = "working";
  protected static final String BROKEN = "broken";


  private ClassifierExample() {}
  

  protected static Datum makeStopLights(String ns, String ew) {
    List<String> features = new ArrayList<String>();
    // Create the north-south light feature
    features.add("NS=" + ns);
    // Create the east-west light feature
    features.add("EW=" + ew);
    // Create the label
    String label = (ns.equals(ew) ? BROKEN : WORKING);
    return new BasicDatum(features, label);
  }


  public static void main(String[] args) {
    // Create a training set
    List<Datum> trainingData = new ArrayList<Datum>();
    trainingData.add(makeStopLights(GREEN, RED));
    trainingData.add(makeStopLights(GREEN, RED));
    trainingData.add(makeStopLights(GREEN, RED));
    trainingData.add(makeStopLights(RED, GREEN));
    trainingData.add(makeStopLights(RED, GREEN));
    trainingData.add(makeStopLights(RED, GREEN));
    trainingData.add(makeStopLights(RED, RED));
    // Create a test set
    Datum workingLights = makeStopLights(GREEN, RED);
    Datum brokenLights = makeStopLights(RED, RED);
    // Build a classifier factory
    LinearClassifierFactory factory = new LinearClassifierFactory();
    factory.useConjugateGradientAscent();
    // Turn on per-iteration convergence updates
    factory.setVerbose(true);
    //Small amount of smoothing
    factory.setSigma(10.0);
    // Build a classifier
    LinearClassifier classifier = (LinearClassifier) factory.trainClassifier(trainingData);
    // Check out the learned weights
    classifier.dump();
    // Test the classifier
    System.out.println("Working instance got: " + classifier.classOf(workingLights));
    classifier.justificationOf(workingLights);
    System.out.println("Broken instance got: " + classifier.classOf(brokenLights));
    classifier.justificationOf(brokenLights);
  }
}
