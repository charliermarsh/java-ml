import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.LinkedList;


public class MultiLayerNeuralNet implements Classifier {
	/* algorithm's learning rate. */
	private final double learning_rate = 1.5;
	/* weights[i][j] is the weight on edge from node i -> j. */
	private final double[][] weights;
	/* out_edges[i] is a list of edges from i. */
	private final LinkedList<Integer>[] out_edges;
	/* in_edges[i] is a list of edges to i. */
	private final LinkedList<Integer>[] in_edges;
	/* layer[i] is a list of nodes in layer i. */
	private final LinkedList<Integer>[] layer;
	/* data set on which to make predictions. */
	private final DataSet d;
	/* number of attributes in data set. */
	private final int N;
	/* number of nodes in the network. */
	private final int num_nodes;
	
	/** Runs an activation threshold function g on some
	 * input value d.
	 */
	private double g(double d) {
		return 1.0/(1.0 + Math.exp(-d));
	}
	
	/** Runs an activation threshold function g's derivative 
	 * on some input value d.
	 */
	private double g_prime(double d) {
		double g = g(d);
		return g * (1.0 - g);
	}
	
	/** Trains the neural network on an example ex by using
	 * the back propagation technique to adjust the net's weights.
	 * Example ex is known to be of classification label.
	 */
	private void back_prop(int[] ex, int label) {		
		double[] a = new double[this.num_nodes];
		double[] in = new double[this.num_nodes];
		double[] delta = new double[this.num_nodes];
		
		// First N nodes are input nodes
		for (int i = 0; i < this.N; i++)
			a[i] = ex[i];
		
		// Compute outputs by propagating inputs forward
		for (int l = 1; l < this.layer.length; l++) {
			for (int dest : this.layer[l]) {
				for (int src : this.in_edges[dest]) {
					in[dest] += this.weights[src][dest]*a[src];
				}
				a[dest] = g(in[dest]);
			}
		}
		if ((int)Math.round(a[this.num_nodes - 1]) == label) return;
		
		// Compute deltas by propagating backward
		// Begin with delta of output layer
		delta[this.num_nodes - 1] = 
				g_prime(in[this.num_nodes - 1])*(label - a[this.num_nodes - 1]); 
		for (int l = this.layer.length - 1; l > 0; l--) {
			for (int src : this.layer[l]) {
				double sum = 0;
				for (int dest : this.out_edges[src]) {
					sum += this.weights[src][dest]*delta[dest];
				}
				delta[src] = g_prime(in[src])*sum;
			}
		}
		
		// Adjust weights
		for (int i = 0; i < this.weights.length; i++) {
			for (int j = 0; j < this.weights.length; j++) {
				if (this.weights[i][j] == 0) continue;
				this.weights[i][j] += this.learning_rate*a[i]*delta[j];
			}
		}
	}
	
	/** Constructor for the MultiLayerNeuralNet class that 
	 * creates a multi-layer, feed-forward neural network 
	 * from a data set.
	 */
	public MultiLayerNeuralNet(DataSet d) {
		this.d = d;
		this.N = this.d.numAttrs;
		int num_hidden = 4;
		this.num_nodes = this.N + num_hidden + 1;
		this.weights = new double[this.num_nodes][this.num_nodes];
		this.in_edges = (LinkedList<Integer>[]) new LinkedList[this.num_nodes];
		this.out_edges = (LinkedList<Integer>[]) new LinkedList[this.num_nodes];
		this.layer = (LinkedList<Integer>[]) new LinkedList[3];

		/* Create first layer and links to hidden layer. */
		this.layer[0] = new LinkedList<Integer>();
		for (int i = 0; i < this.N; i++) {
			this.layer[0].add(i);
			this.in_edges[i] = new LinkedList<Integer>();
			this.out_edges[i] = new LinkedList<Integer>();
			for (int j = this.N; j < this.N + num_hidden; j++) {
				this.weights[i][j] = 5.0 - 10.0*Math.random();
				this.out_edges[i].add(j);
				
				if (this.in_edges[j] == null) {
					this.in_edges[j] = new LinkedList<Integer>(); 
				}
				this.in_edges[j].add(i);
			}
		}
		
		/* Create second layer and links to third layer. */
		this.in_edges[this.num_nodes - 1] = new LinkedList<Integer>();
		this.out_edges[this.num_nodes - 1] = new LinkedList<Integer>();
		this.layer[1] = new LinkedList<Integer>();
		for (int j = this.N; j < this.N + num_hidden; j++) {
			this.layer[1].add(j);
			this.weights[j][this.num_nodes - 1] = 5.0 - 10.0*Math.random();
			
			this.in_edges[this.num_nodes - 1].add(j);
			
			if (this.out_edges[j] == null) {
				this.out_edges[j] = new LinkedList<Integer>(); 
			}
			this.out_edges[j].add(this.num_nodes - 1);
		}
		
		/* Create list of third layer (output node). */ 
		this.layer[2] = new LinkedList<Integer>();
		this.layer[2].add(this.num_nodes - 1);
		
		/* Train on the data set. */
		for (int t = 0; t < 5000; t++) {
			for (int i = 0; i < this.d.numTrainExs; i++) {
				back_prop(this.d.trainEx[i], this.d.trainLabel[i]);
			}
		}
	}

    /** A method for predicting the label of a given example <tt>ex</tt>
     * represented, as in the rest of the code, as an array of values
     * for each of the attributes.  The method should return a
     * prediction, i.e., 0 or 1.
     */
    public int predict(int[] ex) {
    	double[] a = new double[this.num_nodes];
		double[] in = new double[this.num_nodes];
		
		// First N nodes are input nodes
		for (int i = 0; i < this.N; i++)
			a[i] = ex[i];
		
    	// Compute outputs by propagating inputs forward
    	for (int l = 1; l < this.layer.length; l++) {
    		for (int dest : this.layer[l]) {
    			for (int src : this.in_edges[dest]) {
    				in[dest] += this.weights[src][dest]*a[src];
    			}
    			a[dest] = g(in[dest]);
   			}
    	}
    	    	
    	// Return based on output of sigmoid function
    	return (int)Math.round(a[this.num_nodes - 1]);
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
