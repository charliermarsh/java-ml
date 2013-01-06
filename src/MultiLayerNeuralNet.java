import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.LinkedList;

public class MultiLayerNeuralNet implements Classifier {
	/* algorithm's learning rate. */
	private final double learning_rate = 0.5;
	/* weights[i][j] is the weight on edge from node i -> j. 
	 * as a special case, weights[i][i] is the threshold value
	 * of the perceptron i. */
	private final double[][] weights;
	/* outgoingEdges[i] is a list of edges from i. */
	private final LinkedList<Integer>[] outgoingEdges;
	/* incomingEdges[i] is a list of edges to i. */
	private final LinkedList<Integer>[] incomingEdges;
	/* layer[i] is a list of nodes in layer i. */
	private final LinkedList<Integer>[] layer;
	/* data set on which to make predictions. */
	private final DataSet d;
	/* number of attributes in data set. */
	private final int N;
	/* number of nodes in the network. */
	private final int numNodes;
	
	/** Calculates the error on the training examples of
	 * data set d. */
	private double error(DataSet d) {
		if (d.numTrainExs == 0) return 0.0;
		double sum = 0.0;
		for (int i = 0; i < d.numTrainExs; i++)
			sum += Math.abs(d.trainLabel[i] - predict(d.trainEx[i]));
		return sum/d.numTrainExs;
	}
	
	/** Runs an activation threshold function g on some
	 * input value d.
	 */
	private double g(double d) {
		return 1.0/(1.0 + Math.exp(-d));
	}
	
	/** Runs an activation threshold function g's derivative 
	 * on some input value d.
	 */
	private double gPrime(double d) {
		double g = g(d);
		return g * (1.0 - g);
	}
	
	/**
	 * Trains the neural network on every example in data set d.
	 */
	private void backProp(DataSet d) {
		for (int i = 0; i < d.numTrainExs; i++)
			backProp(d.trainEx[i], d.trainLabel[i]);
	}
	
	/** Trains the neural network on an example ex by using
	 * the back propagation technique to adjust the net's weights.
	 * Example ex is known to be of classification label.
	 */
	private void backProp(int[] ex, int label) {
		// output of each node
		double[] a = new double[this.numNodes];
		// input to each node
		double[] in = new double[this.numNodes];
		// delta for adjusting each edge weight
		double[] delta = new double[this.numNodes];
		
		// first N nodes are input nodes
		for (int i = 0; i < this.N; i++)
			a[i] = ex[i];
		
		// compute outputs by propagating inputs forward
		for (int l = 1; l < this.layer.length; l++) {
			for (int dest : this.layer[l]) {
				for (int src : this.incomingEdges[dest]) {
					in[dest] += this.weights[src][dest]*a[src];
				}
				// subtract threshold value
				in[dest] -= this.weights[dest][dest];
				a[dest] = g(in[dest]);
			}
		}
		// return if accurate prediction
		if (predict(a[this.numNodes - 1]) == label) return;
		
		// compute deltas by propagating backward
		// degin with delta of output layer as base case
		delta[this.numNodes - 1] = 
				gPrime(a[this.numNodes - 1])*(label - (int)Math.round(a[this.numNodes - 1])); 
		for (int l = this.layer.length - 2; l >= 0; l--) {
			for (int src : this.layer[l]) {
				double sum = 0;
				for (int dest : this.outgoingEdges[src]) {
					sum += this.weights[src][dest]*delta[dest];
				}
				delta[src] = gPrime(a[src])*sum;
			}
		}
		
		// adjust weights
		for (int i = 0; i < this.weights.length; i++) {
			for (int j = 0; j < this.weights.length; j++) {
				if (this.weights[i][j] == 0) continue;
				this.weights[i][j] += this.learning_rate*a[i]*delta[j];
			}
		}
	}
	
	/** Returns a random weight for an edge. */
	private double randomWeight() {
		return 0.5 - Math.random();
	}
	
	
	/** Resets the weights of a neural network to avoid getting caught
	 * in a local minimum.
	 */
	private void randomizeWeights() {
		for (int i = 0; i < this.weights.length; i++) {
			for (int j = i+1; j < this.weights.length; j++) {
				this.weights[i][j] = randomWeight();
			}
		}
	}
	
	/** Constructor for the MultiLayerNeuralNet class that 
	 * creates a multi-layer, feed-forward neural network 
	 * from a data set.
	 */
	@SuppressWarnings("unchecked")
	public MultiLayerNeuralNet(DataSet d) {		
		this.d = d;
		this.N = this.d.numAttrs;
		// number of nodes in hidden layer
		int numHidden = this.N;
		this.numNodes = this.N + numHidden + 1;
		this.weights = new double[this.numNodes][this.numNodes];
		randomizeWeights();
		// create and initialize list of edges
		this.incomingEdges = (LinkedList<Integer>[]) new LinkedList[this.numNodes];
		this.outgoingEdges = (LinkedList<Integer>[]) new LinkedList[this.numNodes];
		for (int i = 0; i < this.numNodes; i++) {
			this.incomingEdges[i] = new LinkedList<Integer>();
			this.outgoingEdges[i] = new LinkedList<Integer>();
		}
		// number of layers to be included
		int numLayers = 3;
		this.layer = (LinkedList<Integer>[]) new LinkedList[numLayers];
		for (int i = 0; i < numLayers; i++)
			this.layer[i] = new LinkedList<Integer>();

		/* Create first layer and links to hidden layer. */
		for (int i = 0; i < this.N; i++) {
			this.layer[0].add(i);
			// add incoming and outgoing edges
			for (int j = this.N; j < this.N + numHidden; j++) {
				this.outgoingEdges[i].add(j);
				this.incomingEdges[j].add(i);
			}
		}
		
		/* Create second layer and links to third layer. */
		for (int i = this.N; i < this.N + numHidden; i++) {
			this.layer[1].add(i);
			this.incomingEdges[this.numNodes - 1].add(i);
			this.outgoingEdges[i].add(this.numNodes - 1);
		}
		
		/* Create list of third layer (output node). */ 
		this.layer[2].add(this.numNodes - 1);

		// train neural net on each training example
		// run until epsilon threshold error is breached
		double epsilon = 0.05;
		double lastError = Double.MAX_VALUE;
		int maxRuns = 100;
		for (int runs = 0; runs < maxRuns; runs++) {
			backProp(this.d);
			double error = error(this.d);
			if (error < epsilon) 
				break;
			if (error >= lastError) {
				randomizeWeights();
				lastError = Double.MAX_VALUE;
			}
			else
				lastError = error;
		}
	}

    /** A method for predicting the label of a given example <tt>ex</tt>
     * represented, as in the rest of the code, as an array of values
     * for each of the attributes.  The method should return a
     * prediction, i.e., 0 or 1.
     */
    public int predict(int[] ex) {
    	double[] a = new double[this.numNodes];
		double[] in = new double[this.numNodes];
		
		// First N nodes are input nodes
		for (int i = 0; i < this.N; i++)
			a[i] = ex[i];
		
    	// Compute outputs by propagating inputs forward
    	for (int l = 1; l < this.layer.length; l++) {
    		for (int dest : this.layer[l]) {
    			for (int src : this.incomingEdges[dest]) {
    				in[dest] += this.weights[src][dest]*a[src];
    			}
    			a[dest] = g(in[dest]);
   			}
    	}
    	    	
    	// Return based on output of sigmoid function
    	return predict(a[this.numNodes - 1]);
    }
    
    /** Makes a prediction based on some input value a, which
     * should--in practice--be the output value of the final
     * perceptron.
     */
    private int predict(double a) {
    	return (int)Math.round(a);
    }

    /** This method should return a very brief but understandable
     * description of the learning algorithm that is being used,
     * appropriate for posting on the class website.
     */
    public String algorithmDescription() {
    	return "A multi layer neural network.";
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

    	Classifier c = new MultiLayerNeuralNet(d);

    	d.printTestPredictions(c, filestem);
    }
}
