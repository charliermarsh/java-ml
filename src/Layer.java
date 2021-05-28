import java.util.LinkedList;
import java.util.Arrays;

public class Layer {
	/* weights[i][j] is the weight on edge from node i -> j. 
	 * as a special case, weights[i][i] is the threshold value
	 * of the perceptron i. */
	private double[][] weights;
	private double[][] bestWeights;
	/* outgoingEdges[i] is a list of edges from i. */
	private final LinkedList<Integer>[] outgoingEdges;
	/* incomingEdges[i] is a list of edges to i. */
	private final LinkedList<Integer>[] incomingEdges;
	/* number of nodes in the network. */
	private final int numNodes;
	private final int prevNumNodes;
	private final int nextNumNodes;
	
	public Layer(int numNodes, int prevNumNodes, int nextNumNodes) {
		this.numNodes = numNodes;
		this.prevNumNodes = prevNumNodes;
		this.nextNumNodes = nextNumNodes;
		// create and initialize list of edges
		this.incomingEdges = (LinkedList<Integer>[]) new LinkedList[this.numNodes];
		this.outgoingEdges = (LinkedList<Integer>[]) new LinkedList[this.numNodes];
		initEdges();
		linkPrevLayer(0);
		linkNextLayer(0);
		
		this.weights = new double[this.numNodes][this.nextNumNodes];
		this.bestWeights = new double[this.numNodes][this.nextNumNodes];
		randomizeWeights();
	}

	private void initEdges() {
		for (int i = 0; i < this.numNodes; i++) {
			this.incomingEdges[i] = new LinkedList<Integer>();
			this.outgoingEdges[i] = new LinkedList<Integer>();
		}
	}
	
	private void linkPrevLayer(int prevStartNodeNum) { // incoming
		for(int dest = 0; dest < this.numNodes; dest++) {
			for(int src = prevStartNodeNum; src < prevStartNodeNum + this.prevNumNodes; src++) {
				this.incomingEdges[dest].add(src);
			}
		}
	}
	
	private void linkNextLayer(int nextStartNodeNum) { // outgoing
		for(int src = 0; src < this.numNodes; src++) {
			for(int dest = nextStartNodeNum; dest < nextStartNodeNum + this.nextNumNodes; dest++) {
				this.outgoingEdges[src].add(dest);
			}
		}
	}

	public void setWeight(int src, int dest, double weight) {
		this.weights[src][dest] = weight;
	}
	
	public double getWeight(int src, int dest) {
		return this.weights[src][dest];
	}
	
	public LinkedList<Integer> getIncomingEdges(int dest) {
		return this.incomingEdges[dest];
	}
	
	public LinkedList<Integer> getOutgoingEdges(int src){
		return this.outgoingEdges[src];
	}
	
	public int getNumNodes() {
		return this.numNodes;
	}
	
	/** Returns a random weight for an edge. */
	private double randomWeight() {
		return 0.5 - Math.random();
	}
	
	/** Resets the weights of a neural network to avoid getting caught
	 * in a local minimum.
	 */
	public void randomizeWeights() {
		for (int i = 0; i < this.numNodes; i++) {
			for (int j = 0; j < this.nextNumNodes; j++) {
				this.weights[i][j] = randomWeight();
			}
		}
	}
	
	public void assignBestWeights() {
		this.bestWeights = this.weights;
	}
	public void assignWeights() {
		this.weights = this.bestWeights;
	}
	public void updateWeights() {
		for (int i = 0; i < this.weights.length; i++)
			System.arraycopy(this.weights[i], 0, this.bestWeights[i], 0, this.weights[i].length);
	}
}
