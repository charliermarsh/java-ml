import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Random;

public class kNN implements Classifier {
	
	// data set of training examples
	private DataSet dataSet;
	// minimum possible value of k
	private static final int kMin = 1;
	// maximum possible value of k
	private static final int kMax = 15;
	// cross-validated, optimized value of k
	private int kOpt = 7;
	// elimAttr[i] is true if attribute i has been eliminated
	private boolean[] isEliminatedAttr;
	// learning rate for weight training
	private static final double learningRate = 0.05;
	// instanceWeights for training examples
	private double[] instanceWeights;

	
	/** Constructor for the kNN machine learning algorithm.
	 *  Takes as argument a data set. From then on, examples
	 *  in the data set can be fed to predict() in return for
	 *  classifications.
	 */
	public kNN(DataSet dataSet) {
		/* Setup array labelledData so that it contains all the training
		   data attributes along with that example's label. */
		this.dataSet = dataSet;
		this.isEliminatedAttr = new boolean[this.dataSet.numAttrs];
		this.instanceWeights = new double[this.dataSet.numTrainExs];
		this.kOpt = findOptimalK(this.kMin, this.kMax);
		
		initInstanceWeights();
		backwardsElimination();
		traininstanceWeights(100000);		
	}

	private void initInstanceWeights() {
		for (int i = 0; i < this.instanceWeights.length; i++)
			this.instanceWeights[i] = 1.0;
	}
	
	/** Constructor for the kNN machine learning algorithm.
	 *  Mainly used for testing the weight training heuristic.
	 */
	public kNN(DataSet dataSet, int kOpt, int numIteration) {
		/* Setup array labelledData so that it contains all the training
		   data attributes along with that example's label. */
		this.dataSet = dataSet;
		this.isEliminatedAttr = new boolean[this.dataSet.numAttrs];
		this.instanceWeights = new double[this.dataSet.numTrainExs];
		this.kOpt = kOpt;
		
		initInstanceWeights();
		traininstanceWeights(numIteration);		
	}
	
	/** Constructor for the kNN machine learning algorithm.
	 *  Takes as argument a data set and two indices, ignoring
	 *  training examples between those indices 
	 *  to allow for cross validation. Additionally, takes 
	 *  an array of eliminated attributes, instanceWeights for each training
	 *  example, and an optimal k value. From then on, examples
	 *  in the data set can be fed to predict() in return for
	 *  classifications. This constructor is used when a kNN
	 *  instance is required for which no optimizations should
	 *  be performed.
	 */
	public kNN(DataSet dataSet, int from, int to, int kOpt, 
			boolean[] isEliminatedAttr, double[] instanceWeights) {
		/* Setup array labelledData so that it contains all the training
		   data attributes along with that example's label. */
		// create data set, excluding firstEx to lastEx examples
		DataSet subset = initSubset(dataSet, from, to);
		
		this.dataSet = subset;
		this.kOpt = kOpt;
		this.isEliminatedAttr = isEliminatedAttr;
		this.instanceWeights = instanceWeights;	
	}

	private DataSet initSubset(DataSet dataSet, int from, int to) {
		DataSet subset = new DataSet();
		subset.numAttrs = dataSet.numAttrs;
		subset.numTrainExs = dataSet.numTrainExs - (to - from);
		subset.trainEx = new int[subset.numTrainExs][subset.numAttrs];
		subset.trainLabel = new int[subset.numTrainExs];
		
		initSubTrainData(dataSet, from, to, subset);
		return subset;
	}

	private void initSubTrainData(DataSet dataSet, int from, int to, DataSet subset) {
		for (int i = 0; i < from; i++) {
			subset.trainEx[i] = dataSet.trainEx[i];
			subset.trainLabel[i] = dataSet.trainLabel[i];
		}
		for (int i = to; i < dataSet.numTrainExs; i++) {
			subset.trainEx[i - to + from] = dataSet.trainEx[i];
			subset.trainLabel[i - to + from] = dataSet.trainLabel[i];
		}
	}
	
	/** Computes the squared distance between two integer
	 * vectors a and b.
	 */
	private double calcDistance(int[] vector1, int[] vector2) {
		int len = Math.min(vector1.length, vector2.length);
		int distance = 0;
		for (int i = 0; i < len; i++) {
			// skip if attribute is eliminated
			if (this.isEliminatedAttr[i] == true) continue;
			distance += Math.abs(vector1[i] - vector2[i]);
		}
		return distance;
	}
	
	/** Calculates the error over a labeled data set, returning
	 * a double that represents the percent error.
	 */
	private double calcErrorWithCrossValidation() {
		double error = 0.0;
		
		// use 8 different sets for cross validation
		int numSets = 8;
		for (int setNum = 0; setNum < numSets; setNum++) {
			int from = setNum*this.dataSet.numTrainExs/numSets;
			int to = (setNum+1)*this.dataSet.numTrainExs/numSets;
			
			// create new kNN using subset of data set
			kNN knn = new kNN(this.dataSet, from, to, this.kOpt,
					this.isEliminatedAttr, this.instanceWeights);
			
			for (int i = from; i < to; i++) {
				boolean isWrongPredict = knn.predict(this.dataSet.trainEx[i]) != this.dataSet.trainLabel[i];
				if (isWrongPredict)
					error++;
			}
		}
		return error;
	}
	
	/** Calculates the error over a labeled data set using pre-computed
	 * set of indices a in which a[i][j] represents index in training
	 * set d of the jth closest example to i. Returns a double that 
	 * represents the percent error.
	 */
	private double calcError(int[][] kNNindices) {
		double error = 0.0;
		for (int i = 0; i < this.dataSet.numTrainExs; i++) {
			boolean isWrongPredict = voteCount(kNNindices[i]) != this.dataSet.trainLabel[i];
			if (isWrongPredict)
				error++;
		}
		return error;
	}
	
	/** Trains the instanceWeights of the attributes using backwards
	 * propagation on data set d, running T iterations.
	 */
	private void traininstanceWeights(int numIteration) {	
		// get k nearest indices for each training example
		// as determined by cross validation
		int[][] kNNIndicesSet = kNearestWithCrossValidation();
		
		// run T iterations of weight training
		for (int t = 0; t < numIteration; t++) {

			// alter instanceWeights on each example
			for (int i = 0; i < this.dataSet.numTrainExs; i++) {
				
				// modify instanceWeights to satisfy example
				boolean isWrongPredict = this.dataSet.trainLabel[i] != voteCount(kNNIndicesSet[i]);
				while (isWrongPredict) {
					for (int k = 0; k < this.kOpt; k++) {
						int neighborIndex = kNNIndicesSet[i][k];
						if (this.dataSet.trainLabel[neighborIndex] != this.dataSet.trainLabel[i])
							this.instanceWeights[neighborIndex] -= this.learningRate;
						else
							this.instanceWeights[neighborIndex] += this.learningRate;
					}
				}
			}
		}
		//System.out.println("instanceWeights trained.");
	}

	private int[][] kNearestWithCrossValidation() {
		int[][] kNNIndicesSet = new int[this.dataSet.numTrainExs][this.kOpt];

		// use 8 different sets for cross validation
		int numSets = 8;
		for (int setNum = 0; setNum < numSets; setNum++) {
			int from = setNum*this.dataSet.numTrainExs/numSets;
			int to = (setNum+1)*this.dataSet.numTrainExs/numSets;
			
			// create new kNN using subset of data set
			kNN knn = new kNN(this.dataSet, from, to, this.kOpt,
					this.isEliminatedAttr, this.instanceWeights);
			
			for (int i = from; i < to; i++)
				kNNIndicesSet[i] = 
					knn.kNearest(this.kOpt, this.dataSet.trainEx[i]);
		}
		return kNNIndicesSet;
	}
	
	/** Uses backwards elimination to remove attributes from consideration
	 * that decrease the classifier's performance. To avoid recomputing
	 * distances, employs a linear-time distance update that just alters
	 * pre-computed distances based on attribute in question.
	 */
	private void backwardsElimination() {
		// calculate all distances to avoid recomputation
		double[][] dists = new double[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
		
		// use 8 different sets for cross validation (set dist to infinity)
		int numSets = 8;
		for (int setNum = 0; setNum < numSets; setNum++) {
			int from = setNum*this.dataSet.numTrainExs/numSets;
			int to = (setNum+1)*this.dataSet.numTrainExs/numSets;

			for (int t = from; t < to; t++) {
				for (int s = t+1; s < to; s++) {
					dists[t][s] = Double.POSITIVE_INFINITY;
					dists[s][t] = dists[t][s];
				}
			}
		}

		for (int i = 0; i < dists.length; i++) {
			for (int j = i+1; j < dists.length; j++) {
				if (dists[i][j] == Double.POSITIVE_INFINITY) continue;
				
				dists[i][j] = calcDistance(this.dataSet.trainEx[i], this.dataSet.trainEx[j]);
				dists[j][i] = dists[i][j];
			}
		}
		
		// orderedIndices[i][j] is index of jth closest example to i
		int[][] orderedIndices = new int[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
		for (int i = 0; i < orderedIndices.length; i++) {
			// annoying Integer to int casting issues
			Integer[] a = new Integer[this.dataSet.numTrainExs];
			for (int j = 0; j < orderedIndices.length; j++) {
				a[j] = j;
			}
			exComparator comp = new exComparator(dists[i], i);
			comp.isDescending = true;
			Arrays.sort(a, comp);
			for (int j = 0; j < orderedIndices.length; j++) {
				orderedIndices[i][j] = a[j];
			}
		}
		
		// calculate base error with no attribute elimination
		double baselineError = calcError(orderedIndices);
		
		int sum = 0;
		// iterate over each attribute
		for (int m = 0; m < this.dataSet.numAttrs; m++) {
			double[][] newDists = new double[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
			
			// linear-time distance update
			for (int i = 0; i < dists.length; i++) {
				for (int j = i+1; j < dists.length; j++) {
					newDists[i][j] = dists[i][j] - 
							Math.abs(this.dataSet.trainEx[i][m] - this.dataSet.trainEx[j][m]);
					newDists[j][i] = newDists[i][j];
				}
			}
			
			// compute new k nearest
			for (int i = 0; i < this.dataSet.numTrainExs; i++) {
				// annoying Integer to int casting issues
				Integer[] a = new Integer[this.dataSet.numTrainExs];
				for (int j = 0; j < orderedIndices.length; j++) {
					a[j] = orderedIndices[i][j];
				}
				exComparator comp = new exComparator(newDists[i], i);
				comp.isDescending = true;
				Arrays.sort(a, comp);
				for (int j = 0; j < orderedIndices.length; j++) {
					orderedIndices[i][j] = a[j];
				}
			}

			double adjustedError = calcError(orderedIndices);

			// if error improved, keep attribute eliminated; else, retain
			if (adjustedError < baselineError) {
				this.isEliminatedAttr[m] = true;
				baselineError = adjustedError;
				dists = newDists;
				sum++;
			}
		}
		//System.out.printf("%d attributes removed.\n", sum);
	}
	
	/** Uses forward selection to add attributes to a distance function,
	 * greedily adding the attribute that minimizes error.
	 */
	private void forwardsSelection() {
		// remove all attributes
		for (int i = 0; i < this.isEliminatedAttr.length; i++)
			this.isEliminatedAttr[i] = true;
		
		// calculate all distances to avoid recomputation
		double[][] dists = new double[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
		
		// use 8 different sets for cross validation (set dist to infinity)
		int numSets = 8;
		for (int setNum = 0; setNum < numSets; setNum++) {
			int from = setNum*this.dataSet.numTrainExs/numSets;
			int to = (setNum+1)*this.dataSet.numTrainExs/numSets;

			for (int t = from; t < to; t++) {
				for (int s = t+1; s < to; s++) {
					dists[t][s] = Double.POSITIVE_INFINITY;
					dists[s][t] = dists[t][s];
				}
			}
		}
		
		// orderedIndices[i][j] is index of jth closest example to i
		int[][] orderedIndices = new int[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
		for (int i = 0; i < orderedIndices.length; i++) {
			// annoying Integer to int casting issues
			Integer[] a = new Integer[this.dataSet.numTrainExs];
			for (int j = 0; j < orderedIndices.length; j++) {
				a[j] = j;
			}
			exComparator comp = new exComparator(dists[i], i);
			comp.isDescending = true;
			Arrays.sort(a, comp);
			for (int j = 0; j < orderedIndices.length; j++) {
				orderedIndices[i][j] = a[j];
			}
		}
		
		// calculate base error with no attribute elimination
		double baselineError = calcError(orderedIndices);
		boolean attributeAdded;
		
		do {
			attributeAdded = false;
			double minError = Double.POSITIVE_INFINITY;
			int minErrorIndex = -1;
			double[][] minErrorDists = new double[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
			
			// iterate over each attribute
			for (int m = 0; m < this.dataSet.numAttrs; m++) {
				if (!this.isEliminatedAttr[m]) continue;
				
				double[][] newDists = new double[this.dataSet.numTrainExs][this.dataSet.numTrainExs];

				// linear-time distance update
				for (int i = 0; i < dists.length; i++) {
					for (int j = i+1; j < dists.length; j++) {
						newDists[i][j] = dists[i][j] + 
								Math.abs(this.dataSet.trainEx[i][m] - this.dataSet.trainEx[j][m]);
						newDists[j][i] = newDists[i][j];
					}
				}

				// compute new k nearest
				for (int i = 0; i < this.dataSet.numTrainExs; i++) {
					// annoying Integer to int casting issues
					Integer[] a = new Integer[this.dataSet.numTrainExs];
					for (int j = 0; j < orderedIndices.length; j++) {
						a[j] = orderedIndices[i][j];
					}
					exComparator comp = new exComparator(newDists[i], i);
					comp.isDescending = true;
					Arrays.sort(a, comp);
					for (int j = 0; j < orderedIndices.length; j++) {
						orderedIndices[i][j] = a[j];
					}
				}

				double adjustedError = calcError(orderedIndices);

				// if error improved, keep attribute eliminated; else, retain
				if (adjustedError < minError) {
					minError = adjustedError;
					minErrorIndex = m;
					minErrorDists = newDists;
				}
			}
			
			if (minError < baselineError) {
				this.isEliminatedAttr[minErrorIndex] = false;
				dists = minErrorDists;
				attributeAdded = true;
				//System.out.println("Added attribute " + minErrorIndex);
			}
		} while (attributeAdded);
	}
	
	/** Uses a cross-validation technique to find the optimal value of
	 * k for the kNN algorithm on data set d. Returns an integer k 
	 * between k_min and k_max (inclusive).
	 */
	private int findOptimalK(int kMin, int kMax) {
		assert(kMax >= kMin);
		
		int[] kErrors = optimizeK(kMin, kMax);
		// set k to that of minimized error
		double minError = Double.MAX_VALUE;
		int kOpt = 0;
		for (int k = kMin; k <= kMax; k++) {
			if (kErrors[k - kMin] < minError) {
				minError = kErrors[k - kMin];
				kOpt = k;
			}
		}
		//System.out.printf("Optimal k chosen at k = %d\n", minK);
		return kOpt;
	}

	private int[] optimizeK(int kMin, int kMax) {
		// use 8 different sets for cross validation
		int numSets = 8;
		int[] kErrors = new int[kMax - kMin + 1];
		
		for (int setNum = 0; setNum < numSets; setNum++) {
			int from = setNum*this.dataSet.numTrainExs/numSets;
			int to = (setNum+1)*this.dataSet.numTrainExs/numSets;
			
			// create new kNN using subset of data set
			kNN knn = new kNN(this.dataSet, from, to, this.kOpt,
					this.isEliminatedAttr, this.instanceWeights);
			
			// test on held-out training examples
			for (int t = from; t < to; t++) {

				// get k_max best examples
				int[] kNNIndices = knn.kNearest(kMax, this.dataSet.trainEx[t]);

				// count votes by value of k
				double vote_0 = 0;
				double vote_1 = 0;
				for (int k = 0; k < kMax; k++) {
					int neighborIndex = kNNIndices[k];

					// track errors for appropriate k
					if (k >= kMin) {
						int predict = (vote_1 > vote_0)? 1 : 0;
						if (predict != this.dataSet.trainLabel[t]) 
							kErrors[k - kMin]++;
					}

					// continue to increment vote counts
					if (this.dataSet.trainLabel[neighborIndex] == 1)
						vote_1 += this.instanceWeights[neighborIndex];
					else
						vote_0 += this.instanceWeights[neighborIndex];
				}
				
				int predict = (vote_1 > vote_0)? 1 : 0;
				
				if (predict != this.dataSet.trainLabel[t]) 
					kErrors[kMax - kMin]++;
			}
		}
		return kErrors;
	}
	
	/** A class used to modularize comparisons for training
	 * examples based on a specific reference example ex.
	 * ex_index is used to avoid using the same training
	 * example as the reference example.
	 */
	private class exComparator implements Comparator {
		private double[] dists;
	    private int[] ex;
	    private int exIndex = -1;
	    public boolean isDescending = false;
	    
	    private exComparator(int[] ex) {
	      this.ex = ex;
	    }
	    
	    /* Constructor which assumes base ex may be used in comparison */
	    private exComparator(int[] ex, int exIndex) {
		  this.ex = ex;
		  this.exIndex = exIndex;
		}
	    
	    /* Constructor which allows for precomputed distances */
	    private exComparator(double[] dists, int exIndex) {
			this.dists = dists;
			this.exIndex = exIndex;
		}

	    public int compare(Object object1, Object object2) {
	    	int trainExIndex1 = (int)(Integer)object1;
	    	int trainExIndex2 = (int)(Integer)object2;
	    	
	    	// ignore if training example is in data set
	    	if (trainExIndex1 == this.exIndex) return 1;
	    	if (trainExIndex2 == this.exIndex) return -1;
	        
	        // take column of min distance
	    	double dist1;
	    	double dist2;
	    	if (this.dists == null) {
	    	    dist1 = calcDistance(dataSet.trainEx[trainExIndex1], this.ex);
	    		dist2 = calcDistance(dataSet.trainEx[trainExIndex2], this.ex);
	    	}
	    	else {
	    		dist1 = this.dists[trainExIndex1];
	    		dist2 = this.dists[trainExIndex2];	    		
	    	}
	    	int result = 0;
	    	if (dist1 > dist2) result = -1;
    		if (dist2 > dist1) result = 1;
    		if (this.isDescending) result *= -1;
    		return result;
	    }
	}

	/** Calculates the indices of the k nearest training
	 * examples in data set d to example ex. Returns an
	 * array a in which a[i] is the index of the ith
	 * closest training example.
	 */
	private int[] kNearest(int k, int[] ex) {
		// indices of k best examples
		int[] kNNindices = new int[k];
		
		// record distances to avoid recalculation
		double[] dists = new double[this.dataSet.numTrainExs];
		
		// store indices in priority queue, sorted by distance to ex
    	PriorityQueue<Integer> pq = new PriorityQueue<Integer>(k, new exComparator(ex));
    	
    	// search every example
    	for (int i = 0; i < this.dataSet.numTrainExs; i++) {
    		dists[i] = calcDistance(this.dataSet.trainEx[i], ex);
    		if (pq.size() >= k) {
    			if (dists[i] < dists[pq.peek()]) {
    				pq.remove();
    				pq.add(i);
    			}
    		}
    		else {
    			pq.add(i);
    		}
    	}
    	
    	// pq returns worst index first; store backwards in indices
    	for (int i = kNNindices.length - 1; i >= 0; i--)
    		kNNindices[i] = pq.remove();
    	return kNNindices;
	}
	
	/** Counts up the votes for the training examples with labels
	 * at indices listed in array a. Returns 0 or 1.
	 */
	private int voteCount(int[] indices) {
		double vote_1 = 0;
    	double vote_0 = 0;
    	int len = Math.min(indices.length, this.kOpt);
    	for (int k = 0; k < len; k++) {
    		int index = indices[k];
    		if (this.dataSet.trainLabel[index] == 1)
    			vote_1 += this.instanceWeights[index];
    		else
    			vote_0 += this.instanceWeights[index];
    	}
    	
    	return (vote_1 > vote_0)? 1 : 0;
	}
	
    /** A method for predicting the label of a given example <tt>ex</tt>
     * represented, as in the rest of the code, as an array of values
     * for each of the attributes.  The method should return a
     * prediction, i.e., 0 or 1.
     */
    public int predict(int[] ex) {
    	int[] kNNindices = kNearest(this.kOpt, ex);
    	return voteCount(kNNindices);
    }

    /** This method should return a very brief but understandable
     * description of the learning algorithm that is being used,
     * appropriate for posting on the class website.
     */
    public String algorithmDescription() {
    	return "A kNN implementation with cross-validation, backwards elimination, and weight training.";
    }

    /** This method should return the "author" of this program as you
     * would like it to appear on the class website.  You can use your
     * real name, or a pseudonym, or a name that identifies your
     * group.
    */
    public String author() {
    	return "crm & dmrd";
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

	Classifier c = new kNN(d);

	d.printTestPredictions(c, filestem);
    }
}
