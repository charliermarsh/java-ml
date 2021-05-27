import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Random;

public class kNN implements Classifier {
	// commmit test
	
	// data set of training examples
	private DataSet d;
	// minimum possible value of k
	private int kMin = 1;
	// maximum possible value of k
	private int kMax = 15;
	// cross-validated, optimized value of k
	private int kOpt = 7;
	// elimAttr[i] is true if attribute i has been eliminated
	private boolean[] elimAttr;
	// learning rate for weight training
	private double learningRate = 0.05;
	// instanceWeights for training examples
	private double[] instanceWeights;

	
	/** Constructor for the kNN machine learning algorithm.
	 *  Takes as argument a data set. From then on, examples
	 *  in the data set can be fed to predict() in return for
	 *  classifications.
	 */
	public kNN(DataSet d) {
		/* Setup array labelledData so that it contains all the training
		   data attributes along with that example's label. */
		this.d = d;
		this.elimAttr = new boolean[this.d.numAttrs];
		this.instanceWeights = new double[this.d.numTrainExs];
		for (int i = 0; i < this.instanceWeights.length; i++)
			this.instanceWeights[i] = 1.0;
		
		this.kOpt = optimizeK(this.kMin, this.kMax);
		backwardsElimination();
		traininstanceWeights(1);		
	}
	
	/** Constructor for the kNN machine learning algorithm.
	 *  Mainly used for testing the weight training heuristic.
	 */
	public kNN(DataSet d, int kOpt, int T) {
		/* Setup array labelledData so that it contains all the training
		   data attributes along with that example's label. */
		this.d = d;
		this.elimAttr = new boolean[this.d.numAttrs];
		this.instanceWeights = new double[this.d.numTrainExs];
		for (int i = 0; i < this.instanceWeights.length; i++)
			this.instanceWeights[i] = 1.0;
		
		this.kOpt = kOpt;
		traininstanceWeights(T);		
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
	public kNN(DataSet d, int from, int to, int kOpt, 
			boolean[] elimAttr, double[] instanceWeights) {
		/* Setup array labelledData so that it contains all the training
		   data attributes along with that example's label. */
		// create data set, excluding firstEx to lastEx examples
		DataSet subset = new DataSet();
		subset.numAttrs = d.numAttrs;
		subset.numTrainExs = d.numTrainExs - (to - from);
		subset.trainEx = new int[subset.numTrainExs][subset.numAttrs];
		subset.trainLabel = new int[subset.numTrainExs];
		for (int i = 0; i < from; i++) {
			subset.trainEx[i] = d.trainEx[i];
			subset.trainLabel[i] = d.trainLabel[i];
		}
		for (int i = to; i < d.numTrainExs; i++) {
			subset.trainEx[i - to + from] = d.trainEx[i];
			subset.trainLabel[i - to + from] = d.trainLabel[i];
		}
		
		this.d = subset;
		this.kOpt = kOpt;
		this.elimAttr = elimAttr;
		this.instanceWeights = instanceWeights;	
	}
	
	/** Computes the squared distance between two integer
	 * vectors a and b.
	 */
	private double dist(int[] a, int[] b) {
		int len = Math.min(a.length, b.length);
		int sum = 0;
		for (int i = 0; i < len; i++) {
			// skip if attribute is eliminated
			if (this.elimAttr[i]) continue;
			sum += Math.abs(a[i] - b[i]);
		}
		return sum;
	}
	
	/** Calculates the error over a labeled data set, returning
	 * a double that represents the percent error.
	 */
	private double error() {
		double error = 0.0;
		
		// use 8 different sets for cross validation
		int numSets = 8;
		for (int setNum = 0; setNum < numSets; setNum++) {
			int from = setNum*this.d.numTrainExs/numSets;
			int to = (setNum+1)*this.d.numTrainExs/numSets;
			
			// create new kNN using subset of data set
			kNN knn = new kNN(this.d, from, to, this.kOpt,
					this.elimAttr, this.instanceWeights);
			
			for (int t = from; t < to; t++) {
				if (knn.predict(this.d.trainEx[t]) != this.d.trainLabel[t])
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
	private double error(int[][] a) {
		double error = 0.0;
		for (int i = 0; i < this.d.numTrainExs; i++) {
			if (voteCount(a[i]) != this.d.trainLabel[i])
				error++;
		}
		return error;
	}
	
	/** Trains the instanceWeights of the attributes using backwards
	 * propagation on data set d, running T iterations.
	 */
	private void traininstanceWeights(int T) {	
		// get k nearest indices for each training example
		// as determined by cross validation
		int[][] kBest = new int[this.d.numTrainExs][this.kOpt];

		// use 8 different sets for cross validation
		int numSets = 8;
		for (int setNum = 0; setNum < numSets; setNum++) {
			int from = setNum*this.d.numTrainExs/numSets;
			int to = (setNum+1)*this.d.numTrainExs/numSets;
			
			// create new kNN using subset of data set
			kNN knn = new kNN(this.d, from, to, this.kOpt,
					this.elimAttr, this.instanceWeights);
			
			for (int t = from; t < to; t++)
				kBest[t] = 
					knn.kNearest(this.kOpt, this.d.trainEx[t]);
		}
		
		// run T iterations of weight training
		for (int t = 0; t < T; t++) {

			// alter instanceWeights on each example
			for (int i = 0; i < this.d.numTrainExs; i++) {
				
				// modify instanceWeights to satisfy example
				while (this.d.trainLabel[i] != voteCount(kBest[i])) {
					for (int k = 0; k < this.kOpt; k++) {
						int neighborIndex = kBest[i][k];
						if (this.d.trainLabel[neighborIndex] != this.d.trainLabel[i])
							this.instanceWeights[neighborIndex] -= this.learningRate;
						else
							this.instanceWeights[neighborIndex] += this.learningRate;
					}
				}
			}
		}
		//System.out.println("instanceWeights trained.");
	}
	
	/** Uses backwards elimination to remove attributes from consideration
	 * that decrease the classifier's performance. To avoid recomputing
	 * distances, employs a linear-time distance update that just alters
	 * pre-computed distances based on attribute in question.
	 */
	private void backwardsElimination() {
		int numSets=8;
		
		double[][] distances = distanceSettingForBackward(numSets);
		
		int[][] orderedIndices = orderedIndices(distances);
		
		// calculate base error with no attribute elimination
		double baselineError = error(orderedIndices);
		
		distances = iterateEachAttribute(distances, orderedIndices, baselineError);
	}

	private double[][] iterateEachAttribute(double[][] distances, int[][] orderedIndices, double baselineError) {
		int sum = 0;
		double[][] newdistance = new double[this.d.numTrainExs][this.d.numTrainExs];
		int[][] orderedIndice = orderedIndices;
		double baselineerror = baselineError;
		// iterate over each attribute
		for (int m = 0; m < this.d.numAttrs; m++) {
			double[][] temporaryDistances = new double[this.d.numTrainExs][this.d.numTrainExs];
			temporaryDistances = linearDistanceUpdate(distances, m);
			orderedIndice = computeNewNearestK(orderedIndice, temporaryDistances);
			double adjustedError = error(orderedIndice);

			// if error improved, keep attribute eliminated; else, retain
			boolean errorImproved = adjustedError < baselineerror;
			if (errorImproved) {
				this.elimAttr[m] = true;
				baselineerror = adjustedError;
				newdistance = temporaryDistances;
				sum++;
			}
			//if(m == this.d.numAttrs - 1)
			//	System.out.printf("%d attributes removed.\n", sum);
		}
		return newdistance;
	}
	private int[][] computeNewNearestK(int[][] orderedIndice, double[][] temporaryDistances) {
		// compute new k nearest
		int[][] orderedIndices = new int[this.d.numTrainExs][this.d.numTrainExs];
		for (int i = 0; i < this.d.numTrainExs; i++) {
			// annoying Integer to int casting issues
			Integer[] integers = new Integer[this.d.numTrainExs];
			for (int j = 0; j < orderedIndice.length; j++) {
				integers[j] = orderedIndice[i][j];
			}
			exComparator comparator = new exComparator(temporaryDistances[i], i);
			comparator.descending = true;
			Arrays.sort(integers, comparator);
			for (int j = 0; j < orderedIndice.length; j++) {
				orderedIndices[i][j] = integers[j];
			}
		}
		return orderedIndices;
	}

	private double[][] linearDistanceUpdate(double[][] distances, int m) {
		// linear-time distance update
		double[][] temporaryDistances = new double[this.d.numTrainExs][this.d.numTrainExs];
		for (int i = 0; i < distances.length; i++) {
			for (int j = i+1; j < distances.length; j++) {
				temporaryDistances[i][j] = distances[i][j] - 
						Math.abs(this.d.trainEx[i][m] - this.d.trainEx[j][m]);
				temporaryDistances[j][i] = temporaryDistances[i][j];
			}
		}
		return temporaryDistances;
	}

	private int[][] orderedIndices(double[][] distances) {
		// orderedIndices[i][j] is index of jth closest example to i
		int[][] orderedIndices = new int[this.d.numTrainExs][this.d.numTrainExs];
		for (int i = 0; i < orderedIndices.length; i++) {
			// annoying Integer to int casting issues
			Integer[] a = new Integer[this.d.numTrainExs];
			for (int j = 0; j < orderedIndices.length; j++) {
				a[j] = j;
			}
			exComparator comparator = new exComparator(distances[i], i);
			comparator.descending = true;
			Arrays.sort(a, comparator);
			for (int j = 0; j < orderedIndices.length; j++) {
				orderedIndices[i][j] = a[j];
			}
		}
		return orderedIndices;
	}

	private double[][] distanceSettingForBackward(int numSets) {
		double[][] distances = setCrossValidationDistance(numSets);
		for (int i = 0; i < distances.length; i++) {
			for (int j = i+1; j < distances.length; j++) {
				if (distances[i][j] == Double.POSITIVE_INFINITY) continue;
				
				distances[i][j] = dist(this.d.trainEx[i], this.d.trainEx[j]);
				distances[j][i] = distances[i][j];
			}
		}
		return distances;
	}

	private double[][] setCrossValidationDistance(int numSets) {
		// calculate all distances to avoid recomputation
		double[][] distances = new double[this.d.numTrainExs][this.d.numTrainExs];
		
		// use 8 different sets for cross validation (set dist to infinity)
		for (int setNum = 0; setNum < numSets; setNum++) {
			int from = setNum*this.d.numTrainExs/numSets;
			int to = (setNum+1)*this.d.numTrainExs/numSets;

			for (int t = from; t < to; t++) {
				for (int s = t+1; s < to; s++) {
					distances[t][s] = Double.POSITIVE_INFINITY;
					distances[s][t] = distances[t][s];
				}
			}
		}
		return distances;
	}
	
	/** Uses forward selection to add attributes to a distance function,
	 * greedily adding the attribute that minimizes error.
	 */
	private void forwardsSelection() {
		removeAllAttributes();
		int numSets=8;
		double[][] distances = setCrossValidationDistance(numSets);
		int[][] orderedIndices = orderedIndices(distances);
		// calculate base error with no attribute elimination
		double baselineError = error(orderedIndices);
		boolean attributeAdded;
		do {
			attributeAdded = false;
			double minError = Double.POSITIVE_INFINITY;
			int minErrorIndex = -1;
			double[][] minErrorDistances = new double[this.d.numTrainExs][this.d.numTrainExs];
			
			// iterate over each attribute
			for (int m = 0; m < this.d.numAttrs; m++) {
				boolean mthEliminateAttributNotEmpty = !this.elimAttr[m];
				if (mthEliminateAttributNotEmpty) continue;
				
				double[][] newDistances = new double[this.d.numTrainExs][this.d.numTrainExs];
				newDistances = linearDistanceUpdate(distances, m);
				
				computeNewNearestK(orderedIndices, newDistances);

				double adjustedError = error(orderedIndices);

				// if error improved, keep attribute eliminated; else, retain
				boolean errorImproved = (adjustedError < minError);
				if (errorImproved) {
					minError = adjustedError;
					minErrorIndex = m;
					minErrorDistances = newDistances;
				}
			}
			
			boolean baselineErrorImproved = minError < baselineError;
			if (baselineErrorImproved) {
				this.elimAttr[minErrorIndex] = false;
				distances = minErrorDistances;
				attributeAdded = true;
				//System.out.println("Added attribute " + minErrorIndex);
			}
		} while (attributeAdded);
	}
	private double[][] linearForwardDistanceUpdate(double[][] distances, int m) {
		// linear-time distance update
		double[][] temporaryDistances = new double[this.d.numTrainExs][this.d.numTrainExs];
		for (int i = 0; i < distances.length; i++) {
			for (int j = i+1; j < distances.length; j++) {
				temporaryDistances[i][j] = distances[i][j] - 
						Math.abs(this.d.trainEx[i][m] + this.d.trainEx[j][m]);
				temporaryDistances[j][i] = temporaryDistances[i][j];
			}
		}
		return temporaryDistances;
	}

	private void removeAllAttributes() {
		// remove all attributes
		for (int i = 0; i < this.elimAttr.length; i++)
			this.elimAttr[i] = true;
	}
	
	/** Uses a cross-validation technique to find the optimal value of
	 * k for the kNN algorithm on data set d. Returns an integer k 
	 * between k_min and k_max (inclusive).
	 */
	private int optimizeK(int kMin, int kMax) {
		assert(kMax >= kMin);
		
		// use 8 different sets for cross validation
		int numSets = 8;
		int[] kErrors = new int[kMax - kMin + 1];
		
		for (int setNum = 0; setNum < numSets; setNum++) {
			int from = setNum*this.d.numTrainExs/numSets;
			int to = (setNum+1)*this.d.numTrainExs/numSets;
			
			// create new kNN using subset of data set
			kNN knn = new kNN(this.d, from, to, this.kOpt,
					this.elimAttr, this.instanceWeights);
			
			// test on held-out training examples
			for (int t = from; t < to; t++) {

				// get k_max best examples
				int[] best = knn.kNearest(kMax, this.d.trainEx[t]);

				// count votes by value of k
				double vote_0 = 0;
				double vote_1 = 0;
				for (int k = 0; k < kMax; k++) {
					int i = best[k];

					// track errors for appropriate k
					if (k >= kMin) {
						int result = (vote_1 > vote_0)? 1 : 0;
						if (result != this.d.trainLabel[t]) 
							kErrors[k - kMin]++;
					}

					// continue to increment vote counts
					if (this.d.trainLabel[i] == 1)
						vote_1 += this.instanceWeights[i];
					else
						vote_0 += this.instanceWeights[i];
				}
				int result = (vote_1 > vote_0)? 1 : 0;
				if (result != this.d.trainLabel[t]) 
					kErrors[kMax - kMin]++;
			}
		}
		// set k to that of minimized error
		double min = Double.MAX_VALUE;
		int minK = 0;
		for (int k = kMin; k <= kMax; k++) {
			if (kErrors[k - kMin] < min) {
				min = kErrors[k - kMin];
				minK = k;
			}
		}
		//System.out.printf("Optimal k chosen at k = %d\n", minK);
		return minK;
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
	    public boolean descending = false;
	    
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

	    public int compare(Object o1, Object o2) {
	    	int i = (int)(Integer)o1;
	    	int j = (int)(Integer)o2;
	    	
	    	// ignore if training example is in data set
	    	if (i == this.exIndex) return 1;
	    	if (j == this.exIndex) return -1;
	        
	        // take column of min distance
	    	double d1;
	    	double d2;
	    	if (this.dists == null) {
	    	    d1 = dist(d.trainEx[i], this.ex);
	    		d2 = dist(d.trainEx[j], this.ex);
	    	}
	    	else {
	    		d1 = this.dists[i];
	    		d2 = this.dists[j];	    		
	    	}
	    	int result = 0;
	    	if (d1 > d2) result = -1;
    		if (d2 > d1) result = 1;
    		if (this.descending) result *= -1;
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
		int[] indices = new int[k];
		
		// record distances to avoid recalculation
		double[] dists = new double[this.d.numTrainExs];
		
		// store indices in priority queue, sorted by distance to ex
    	PriorityQueue<Integer> pq = new PriorityQueue<Integer>(k, new exComparator(ex));
    	
    	// search every example
    	for (int i = 0; i < this.d.numTrainExs; i++) {
    		dists[i] = dist(this.d.trainEx[i], ex);
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
    	for (int i = indices.length - 1; i >= 0; i--)
    		indices[i] = pq.remove();
    	return indices;
	}
	
	/** Counts up the votes for the training examples with labels
	 * at indices listed in array a. Returns 0 or 1.
	 */
	private int voteCount(int[] a) {
		double vote_1 = 0;
    	double vote_0 = 0;
    	int len = Math.min(a.length, this.kOpt);
    	for (int k = 0; k < len; k++) {
    		int i = a[k];
    		if (this.d.trainLabel[i] == 1)
    			vote_1 += this.instanceWeights[i];
    		else
    			vote_0 += this.instanceWeights[i];
    	}
    	return (vote_1 > vote_0)? 1 : 0;
	}
	
    /** A method for predicting the label of a given example <tt>ex</tt>
     * represented, as in the rest of the code, as an array of values
     * for each of the attributes.  The method should return a
     * prediction, i.e., 0 or 1.
     */
    public int predict(int[] ex) {
    	int[] indices = kNearest(this.kOpt, ex);
    	return voteCount(indices);
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
