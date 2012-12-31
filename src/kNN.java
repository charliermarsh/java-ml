import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Random;

public class kNN implements Classifier {
	
	// data set of training examples
	private DataSet d;
	// minimum possible value of k
	private int kMin = 5;
	// maximum possible value of k
	private int kMax = 15;
	// cross-validated, optimized value of k
	private int kOpt = 7;
	// elimAttr[i] is true if attribute i has been eliminated
	private boolean[] elimAttr;
	// learning rate for weight training
	private double learningRate = 0.05;
	// weights for attributes
	private double[] weights;

	
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
		this.weights = new double[this.d.numTrainExs];
		for (int i = 0; i < this.weights.length; i++)
			this.weights[i] = 1.0;
		
		//this.kOpt = optimizeK(this.d, this.kMin, this.kMax);
		backwardsEliminate(this.d);
		trainWeights(this.d, 100000);		
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
	private double error(DataSet d) {
		double error = 0.0;
		int num_trials = Math.min(100, d.numTrainExs);
		for (int i = 0; i < num_trials; i++) {
			if (predictFromTraining(d.trainEx[i]) != d.trainLabel[i])
				error++;
		}
		return error;
	}
	
	/** Calculates the error over a labeled data set using pre-computed
	 * set of indices a in which a[i][j] represents index in training
	 * set d of the jth closest example to i. Returns a double that 
	 * represents the percent error.
	 */
	private double error(DataSet d, int[][] a) {
		double error = 0.0;
		int num_trials = Math.min(100, d.numTrainExs);
		for (int i = 0; i < num_trials; i++) {
			if (voteCount(a[i]) != d.trainLabel[i])
				error++;
		}
		return error;
	}
	
	/** Trains the weights of the attributes using backwards
	 * propagation on data set d, running T iterations.
	 */
	private void trainWeights(DataSet d, int T) {	
		// get k nearest indices for each training example
		int[][] kBest = new int[d.numTrainExs][this.kOpt];
		int[] order = new int[d.numTrainExs];
		for (int i = 0; i < kBest.length; i++) {
			kBest[i] = kNearestFromTraining(d, this.kOpt, d.trainEx[i]);
			order[i] = i;
		}
		
		// run T iterations of weight training
		for (int t = 0; t < T; t++) {
			/*
	         * Do the Knuth Shuffle!  It sounds like more fun than it is!
	         */
	        //Set seed to constant to get the same result multiple times
	        Random random = new Random();
	        for (int i = 0; i < order.length; i++) {
	            int swap = random.nextInt(order.length - i);
	            int temp = order[swap];
	            order[swap] = order[order.length - i - 1];
	            order[order.length - i - 1] = temp;
	        }
	        
			// alter weights on each example
			for (int i = 0; i < d.numTrainExs; i++) {
				int exIndex = order[i];
				
				// modify weights to satisfy example
				while (d.trainLabel[exIndex] != voteCount(kBest[exIndex])) {
					for (int k = 0; k < this.kOpt; k++) {
						int neighborIndex = kBest[exIndex][k];
						if (d.trainLabel[neighborIndex] != d.trainLabel[exIndex])
							this.weights[neighborIndex] -= this.learningRate;
						else
							this.weights[neighborIndex] += this.learningRate;
					}
				}
			}
		}
		System.out.println("Weights trained.");
	}
	
	/** Uses backwards elimination to remove attributes from consideration
	 * that decrease the classifier's performance. To avoid recomputing
	 * distances, employs a linear-time distance update that just alters
	 * pre-computed distances based on attribute in question.
	 */
	private void backwardsEliminate(DataSet d) {
		// calculate all distances to avoid recomputation
		double[][] dists = new double[d.numTrainExs][d.numTrainExs];
		for (int i = 0; i < dists.length; i++) {
			for (int j = i+1; j < dists.length; j++) {
				dists[i][j] = dist(d.trainEx[i], d.trainEx[j]);
				dists[j][i] = dists[i][j];
			}
		}
		
		// orderedIndices[i][j] is index of jth closest example to i
		int[][] orderedIndices = new int[d.numTrainExs][d.numTrainExs];
		for (int i = 0; i < orderedIndices.length; i++) {
			// annoying Integer to int casting issues
			Integer[] a = new Integer[d.numTrainExs];
			for (int j = 0; j < orderedIndices.length; j++) {
				a[j] = j;
			}
			Arrays.sort(a, new exComparator(dists[i], i));
			for (int j = 0; j < orderedIndices.length; j++) {
				orderedIndices[i][j] = a[j];
			}
		}
		
		// calculate base error with no attribute elimination
		double baselineError = error(d, orderedIndices);
		
		int sum = 0;
		// iterate over each attribute
		for (int m = 0; m < d.numAttrs; m++) {
			// linear-time distance update
			for (int i = 0; i < dists.length; i++) {
				for (int j = i+1; j < dists.length; j++) {
					dists[i][j] -= Math.abs(d.trainEx[i][m] - d.trainEx[j][m]);
					dists[j][i] = dists[i][j];
				}
			}
			
			/* only compute num_trials new kNN to avoid excessive computation;
			 * i.e., error(d, ...) only take num_trials samples, so no reason
			 * to compute beyond that
			 */
			int num_trials = Math.min(100, d.numTrainExs);
			for (int i = 0; i < num_trials; i++) {
				// annoying Integer to int casting issues
				Integer[] a = new Integer[d.numTrainExs];
				for (int j = 0; j < orderedIndices.length; j++) {
					a[j] = orderedIndices[i][j];
				}
				Arrays.sort(a, new exComparator(dists[i], i));
				for (int j = 0; j < orderedIndices.length; j++) {
					orderedIndices[i][j] = a[j];
				}
			}

			double adjustedError = error(d, orderedIndices);
			// if error improved, keep attribute eliminated; else, retain
			if (adjustedError < baselineError) {
				this.elimAttr[m] = true;
				baselineError = adjustedError;
				sum++;
			}
			else {
				// linear-time distance update
				for (int i = 0; i < dists.length; i++) {
					for (int j = i+1; j < dists.length; j++) {
						dists[i][j] += Math.abs(d.trainEx[i][m] - d.trainEx[j][m]);
						dists[j][i] = dists[i][j];
					}
				}
			}
		}
		System.out.printf("%d attributes removed.\n", sum);
	}
	
	/** Uses a cross-validation technique to find the optimal value of
	 * k for the kNN algorithm on data set d. Returns an integer k 
	 * between k_min and k_max (inclusive).
	 */
	private int optimizeK(DataSet d, int kMin, int kMax) {
		assert(kMax >= kMin);
		
		// run 100 trials to find best k
		int numTrials = Math.min(1000, d.numTrainExs);
		int[] kErrors = new int[kMax - kMin + 1];
		for (int t = 0; t < numTrials; t++) {
			// get k_max best examples
			int[] best = kNearestFromTraining(d, kMax, d.trainEx[t]);
			
	    	// count votes by value of k
	    	double vote_0 = 0;
	    	double vote_1 = 0;
	    	for (int k = 0; k < kMax; k++) {
	    		int i = best[k];
	    		
	    		// track errors for appropriate k
	    		if (k >= kMin) {
	    			int result = (vote_1 > vote_0)? 1 : 0;
		    		if (result != d.trainLabel[t]) 
		    			kErrors[k - kMin]++;
	    		}
	    		
	    		// continue to increment vote counts
	    		if (d.trainLabel[i] == 1)
	    			vote_1 += this.weights[i];
	    		else
	    			vote_0 += this.weights[i];
	    	}
	    	int result = (vote_1 > vote_0)? 1 : 0;
    		if (result != d.trainLabel[t]) 
    			kErrors[kMax - kMin]++;
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
		System.out.printf("Optimal k chosen at k = %d\n", minK);
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
	    	if (d1 > d2) return -1;
    		if (d2 > d1) return 1;
    		return 0;
	    }
	}

	/** Calculates the indices of the k nearest training
	 * examples in data set d to example ex. Returns an
	 * array a in which a[i] is the index of the ith
	 * closest training example.
	 */
	private int[] kNearest(DataSet d, int k, int[] ex) {
		// indices of k best examples
		int[] indices = new int[k];
		
		// record distances to avoid recalculation
		double[] dists = new double[d.numTrainExs];
		
		// store indices in priority queue, sorted by distance to ex
    	PriorityQueue<Integer> pq = new PriorityQueue<Integer>(k, new exComparator(ex));
    	
    	// search every example
    	for (int i = 0; i < d.numTrainExs; i++) {
    		dists[i] = dist(d.trainEx[i], ex);
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
	
	/** Calculates the indices of the k nearest training
	 * examples in data set d to example ex, assuming that
	 * ex itself is a training example. Returns an
	 * array a in which a[i] is the index of the ith
	 * closest training example.
	 */
	private int[] kNearestFromTraining(DataSet d, int k, int[] ex) {
		int[] indices = new int[k];
		int[] kNearest = kNearest(d, k+1, ex);
		for (int i = 0; i < indices.length; i++)
			indices[i] = kNearest[i+1];
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
    		if (d.trainLabel[i] == 1)
    			vote_1 += this.weights[i];
    		else
    			vote_0 += this.weights[i];
    	}
    	return (vote_1 > vote_0)? 1 : 0;
	}
	
    /** A method for predicting the label of a given example <tt>ex</tt>
     * represented, as in the rest of the code, as an array of values
     * for each of the attributes.  The method should return a
     * prediction, i.e., 0 or 1.
     */
    public int predict(int[] ex) {
    	int[] indices = kNearest(this.d, this.kOpt, ex);
    	return voteCount(indices);
    }
    
    /** A method for predicting the label of a given example <tt>ex</tt>
     * represented, as in the rest of the code, as an array of values
     * for each of the attributes.  The method should return a
     * prediction, i.e., 0 or 1. Assumes that ex is a member of the
     * training set.
     */
    public int predictFromTraining(int[] ex) {
    	int[] indices = kNearestFromTraining(this.d, this.kOpt, ex);
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
    	return "dmrd + crm";
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
