import java.io.FileNotFoundException;
import java.io.IOException;

public class SingleLayerNeuralNet implements Classifier {
	/* algorithm's learning rate. */
	private final double learning_rate = .1;
	/* weights[i] is the weight from input perceptron i to
	   the output perceptron. */
	private final double[] weights;
	/* data set on which to make predictions. */
	private final DataSet d;
	/* number of attributes in data set. */
	private final int N;
	
	/** Calculates the error on the training examples of
	 * data set d. */
	private double error(DataSet d) {
		double sum = 0.0;
		for (int i = 0; i < d.numTrainExs; i++)
			sum += Math.abs(d.trainLabel[i] - predict(d.trainEx[i]));
		return sum/d.numTrainExs;
	}
	
	/** Trains the neural network on an example ex by using
	 * the back propagation technique to adjust the net's weights.
	 * Example ex is known to be of classification label.
	 */
	private void back_prop(int[] ex, int label) {
		int output = predict(ex);
		if (output == label) return;
		
		for (int i = 0; i < this.N; i++) {
			double delta = this.learning_rate*(label-output)*ex[i];
			this.weights[i] += delta;
		}
	}
	
	/** Constructor for the SingleLayerNeuralNet class that 
	 * creates a single layer neural network from a data set.
	 */
	public SingleLayerNeuralNet(DataSet d) {
		this.d = d;
		this.N = this.d.numAttrs;
		// N+1 perceptrons, with perceptron N = output
		this.weights = new double[this.N];
		double epsilon = 0.05;
		int maxRuns = 100000;
		
		train(epsilon, maxRuns);
	}

	/** train neural net on each training example 
	 */
	private void train(double epsilon, int maxRuns) {
		int runs = 0;
		while (runs < maxRuns && error(this.d) > epsilon) {
			for (int i = 0; i < this.d.numTrainExs; i++)
				back_prop(this.d.trainEx[i], this.d.trainLabel[i]);
			runs++;
		}
	}

    /** A method for predicting the label of a given example <tt>ex</tt>
     * represented, as in the rest of the code, as an array of values
     * for each of the attributes.  The method should return a
     * prediction, i.e., 0 or 1. Example ex is guaranteed to be represented
     * as an array of binary values.
     */
    public int predict(int[] ex) {
    	double sum = 0.0;
    	for (int i = 0; i < this.N; i++)
    		sum += weights[i]*ex[i];
    	if (sum > 0) return 1;
    	return 0;
    }

    /** This method should return a very brief but understandable
     * description of the learning algorithm that is being used,
     * appropriate for posting on the class website.
     */
    public String algorithmDescription() {
    	return "A single layer neural network.";
    }

    /** This method should return the "author" of this program as you
     * would like it to appear on the class website.  You can use your
     * real name, or a pseudonym, or a name that identifies your
     * group.
    */
    public String author() {
    	return "crm";
    }
    
    /** A simple main for testing this algorithm.  This main reads a
     * filestem from the command line, runs the learning algorithm on
     * this dataset, and prints the test predictions to filestem.testout.
     */
    public static void main(String argv[])
    		throws FileNotFoundException, IOException {
    	if (argv.length < 1) {
    		System.err.println("argument: filestem");
    		return;
    	}

    	String filestem = argv[0];

    	DataSet d = new BinaryDataSet(filestem);

    	Classifier c = new SingleLayerNeuralNet(d);

    	d.printTestPredictions(c, filestem);
    }
}
