import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.LinkedList;
import java.util.Arrays;

public class MultiLayerNeuralNet implements Classifier {
	/* algorithm's learning rate. */
	private final double learningRate = 0.5;
	/* algorithm's momentum parameter. */
	private final double momentumFactor = 0.1;
	/* layer[i] is a list of nodes in layer i. */
	private final Layer[] layer;
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
	private double sigmoid(double d) {
		return 1.0/(1.0 + Math.exp(-d));
	}
	
	/** Runs an activation threshold function g's derivative 
	 * on some input value d.
	 */
	private double sigmoidPrime(double d) {
		double g = sigmoid(d);
		return g * (1.0 - g);
	}
	
	/**
	 * Trains the neural network on every example in data set d
	 * using previous deltas prevDelta.
	 */
	private void backPropagation(DataSet d, double[] prevDelta) {
		for (int i = 0; i < d.numTrainExs; i++)
			backPropagation(d.trainEx[i], d.trainLabel[i], prevDelta);
	}
	
	/** Trains the neural network on an example ex by using
	 * the back propagation technique to adjust the net's weights.
	 * Example ex is known to be of classification label. Uses
	 * 0 for all previous deltas.
	 */
	private void backPropagation(int[] ex, int label) {
		backPropagation(ex, label, new double[this.numNodes]);
	}
	
	/** Trains the neural network on an example ex by using
	 * the back propagation technique to adjust the net's weights.
	 * Example ex is known to be of classification label. It is
	 * assumed that prevDelta are the deltas for time t-1, used
	 * to add momentum to the gradient descent calculation. Stores
	 * the calculated deltas in prevDelta for future use.
	 */
	private void backPropagation(int[] ex, int label, double[] prevDelta) {
		// output of each node
		double[] output = new double[this.numNodes];
		// input to each node
		double[] input = new double[this.numNodes];
		// delta for adjusting each edge weight
		double[] delta = new double[this.numNodes];
		
		// first N nodes are input nodes
		for (int i = 0; i < this.N; i++)
			output[i] = ex[i];
		
		forwardPass(output, input);
		// return if accurate prediction
		if (predict(output[this.numNodes - 1]) == label) return;
		
		backwardPass(label, prevDelta, output, delta);
		
		adjustWeights(output, delta);
	}

	private int getIdx(int layerNum, int nodeNum) {
		int idx = 0;
		for(int l=1; l<layerNum; l++) {
			idx += this.layer[layerNum-1].getNumNodes();
		}
		idx = nodeNum;
		
		return idx;
	}
	/** adjust weights
	 */
	private void adjustWeights(double[] output, double[] delta) {
		for(int l=0; l<this.layer.length - 1; l++) {
			for (int i = 0; i < this.layer[l].getNumNodes(); i++) {
				for (int j = 0; j < this.layer[l+1].getNumNodes(); j++) {
					this.layer[l].setWeight( i, j, this.layer[l].getWeight(i, j) + this.learningRate*output[ getIdx(l,i) ]*delta[ getIdx(l+1,j)]);
				}
			}	
		}
	}

	/** compute deltas by propagating backward
		degin with delta of output layer as base case
	 */
	private void backwardPass(int label, double[] prevDelta, double[] output, double[] delta) {
		delta[this.numNodes - 1] = 
				sigmoidPrime(output[this.numNodes - 1])*(label - (int)Math.round(output[this.numNodes - 1])); 
		for (int l = this.layer.length - 2; l >= 0; l--) {
			for (int src =0 ; src < this.layer[l].getNumNodes(); src++) {
				double sum = 0;
				for (int dest : this.layer[l].getOutgoingEdges(src)) {
					sum += this.layer[l].getWeight(src, dest)*delta[ getIdx(l+1,dest) ];
				}
				// compute delta and add momentum factor
				delta[getIdx(l,src)] = sigmoidPrime(output[ getIdx(l,src) ])*sum;
				delta[getIdx(l,src)] += this.momentumFactor*prevDelta[getIdx(l,src)];
				// store momentum for future use
				prevDelta[getIdx(l,src)] = delta[getIdx(l,src)];
			}
		}
	}

	/** compute outputs by propagating inputs forward
	 */
	private void forwardPass(double[] output, double[] input) {
		for (int l = 1; l < this.layer.length; l++) {
			for (int dest = 0; dest < this.layer[l].getNumNodes(); dest++) {
				for (int src : this.layer[l].getIncomingEdges(dest)) {
					input[ getIdx(l,dest) ] += this.layer[l-1].getWeight(src,dest)*output[ getIdx(l-1,src) ];
				}
				// subtract threshold value
				input[ getIdx(l,dest) ] -= this.layer[l-1].getWeight(dest,dest);
				output[ getIdx(l,dest) ] = sigmoid(input[ getIdx(l,dest) ]);
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
		int numInput = this.N;
		int numOutput = 1;
		this.numNodes = this.N + numHidden + 1;
		// number of layers to be included
		int numLayers = 3;
		this.layer = new Layer[numLayers];		
		Layer inputLayer = new Layer(numInput, 0, numHidden);
		Layer hiddenLayer = new Layer(numHidden, numInput, numOutput);
		Layer outputLayer = new Layer(numOutput, numHidden, 0);
		this.layer[0] = inputLayer;
		this.layer[1] = hiddenLayer;
		this.layer[2] = outputLayer;

		train();
	}

	/** train neural net on each training example
		run until epsilon threshold error is breached
	 */
	private void train() {
		double epsilon = 0.05;
		double minError = Double.MAX_VALUE;
		double lastError = Double.MAX_VALUE;
		double[] prevDelta = new double[this.numNodes];
		int maxRuns = 1;
		for (int runs = 0; runs < maxRuns; runs++) {
			// run back prop
			backPropagation(this.d, prevDelta);
			double error = error(this.d);
			// if error is sufficiently low, cut-off
			if (error < epsilon) {
				for(int l=0; l<this.layer.length; l++)
					this.layer[l].assignBestWeights();
				break;
			}
			// if error has not improved, reset
			else if (error >= lastError) {
				for(int l=0; l<this.layer.length; l++)
					this.layer[l].randomizeWeights();
				lastError = Double.MAX_VALUE;
			}
			else {
				lastError = error;
				// if error is best seen, remember weights
				if (error < minError) {
					minError = error;
					for(int l=0; l<this.layer.length; l++)
						this.layer[l].updateWeights();
				}
			}
		}
		// assign permanent weights to the best weights observed
		for(int l=0; l<this.layer.length; l++)
			this.layer[l].assignWeights();
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
    		for (int dest = 0; dest < this.layer[l].getNumNodes(); dest++) {
    			for (int src : this.layer[l].getIncomingEdges(dest)) {
    				in[dest] += this.layer[l-1].getWeight(src,dest)*a[src];
    			}
    			a[dest] = sigmoid(in[dest]);
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
