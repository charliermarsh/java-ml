import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;

public class kNN implements Classifier {
	
	// an array of labelled training examples
	private int[][] labelledData;
	// labelledData[i][labelIndex] is the ith example's label
	private int labelIndex;
	// minimum possible value of k
	private int k_min = 1;
	// maximum possible value of k
	private int k_max = 15;
	// cross-validated, optimized value of k
	private int k_opt;

	/** Computes the squared distance between two numeric
	 * vectors a and b.
	 */
	private double dist(int[] a, int[] b) {
		int len = Math.min(a.length, b.length);
		double sum1 = 0;
		double sum2 = 0;
		double sum3 = 0;
		for (int i = 0; i < len; i++) {
			sum1 += a[i]*b[i];
			sum2 += a[i]*a[i];
			sum3 += b[i]*b[i];
		}
		return 1.0 - (2*sum1/(sum2+sum3));
	}
	
	/** Uses a cross-validation technique to find the optimal value of
	 * k for the kNN algorithm on data set d. Returns an integer k 
	 * between k_min and k_max (inclusive).
	 */
	private int optimizeK(DataSet d) {
		assert(this.k_max > this.k_min);
		
		// run 100 trials to find best k
		int num_trials = 500;
		int[] k_errors = new int[this.k_max - this.k_min + 1];
		for (int t = 0; t < num_trials; t++) {
			// choose random training example to test values of k
			int r = (int)(Math.random()*this.labelledData.length);
			int[] ex = this.labelledData[r];
			
			// sort array by distance to example
	    	Arrays.sort(this.labelledData, new columnComparator(ex));
			
	    	// count votes by value of k
	    	double vote_0 = 0;
	    	double vote_1 = 0;
	    	for (int k = 0; k < this.k_min; k++) {
	    		double dist = dist(this.labelledData[k], ex);
	    		if (this.labelledData[k][this.labelIndex] == 1.0)
	    			vote_1 += 1.0/dist;
	    		else
	    			vote_0 += 1.0/dist;
	    	}
	    	for (int k = this.k_min; k <= this.k_max; k++) {
	    		// track errors for each value of k
	    		int result = (vote_1 > vote_0)? 1 : 0;
	    		if (result != ex[this.labelIndex]) k_errors[k - this.k_min]++;
	    		
	    		// continue to update votes as k increases
	    		double dist = dist(this.labelledData[k], ex);
	    		if (this.labelledData[k][this.labelIndex] == 1.0)
	    			vote_1 += 1.0/dist;
	    		else
	    			vote_0 += 1.0/dist;
	    	}
		}
		// set k to that of minimized error
		double min = Double.MAX_VALUE;
		int min_k = 0;
		for (int k = this.k_min; k <= this.k_max; k++) {
			if (k_errors[k - this.k_min] < min) {
				min = k_errors[k - this.k_min];
				min_k = k;
			}
		}
		return min_k;
	}
	
	/** A class used to modularize a sort comparator for
	 * an array of training examples.
	 */
	private class columnComparator implements Comparator {
	    private int[] columnToSortOn;
	    
	    private columnComparator(int[] columnToSortOn) {
	      this.columnToSortOn = columnToSortOn;
	    }

	    public int compare(Object o1, Object o2) {
	    	// ignore if training example is in data set
	    	if (o1 == this.columnToSortOn) return 1;
	    	if (o2 == this.columnToSortOn) return -1;
	    	
	    	int[] a1 = (int[])o1;
	        int[] a2 = (int[])o2;
	        
	        // take column of min distance
	        double d1 = dist(a1, this.columnToSortOn);
	        double d2 = dist(a2, this.columnToSortOn);
	        if (d1 > d2) return 1;
	        if (d2 > d1) return -1;
	        return 0;
	    }
	}
	
	/** Constructor for the kNN machine learning algorithm.
	 *  Takes as argument a data set. From then on, examples
	 *  in the data set can be fed to predict() in return for
	 *  classifications.
	 *  @param		dataset		the data set on which to train
	 */
	public kNN(DataSet d) {
		/* Setup array labelledData so that it contains all the training
		   data attributes along with that example's label. */
		this.labelIndex = d.numAttrs;
		this.labelledData = new int[d.numTrainExs][this.labelIndex+1];
		
		for (int i = 0; i < this.labelledData.length; i++) {
			for (int j = 0; j < this.labelIndex; j++) {
				this.labelledData[i][j] = d.trainEx[i][j];
			}
			this.labelledData[i][this.labelIndex] = d.trainLabel[i];
		}
		
		this.k_opt = optimizeK(d);
	}

	
    /** A method for predicting the label of a given example <tt>ex</tt>
     * represented, as in the rest of the code, as an array of values
     * for each of the attributes.  The method should return a
     * prediction, i.e., 0 or 1.
     * @param		ex		values representing an example to classify	
     */
    public int predict(int[] ex) {
    	/* Sort array by distance to example. */
    	Arrays.sort(this.labelledData, new columnComparator(ex));
    	
    	double vote_1 = 0;
    	double vote_0 = 0;
    	for (int i = 0; i < this.k_opt; i++) {
    		double dist = dist(ex, this.labelledData[i]);
    		if (this.labelledData[i][this.labelIndex] == 1.0)
    			vote_1 += 1.0/dist;
    		else
    			vote_0 += 1.0/dist;
    	}
    	return (vote_1 > vote_0)? 1 : 0;
    }

    /** This method should return a very brief but understandable
     * description of the learning algorithm that is being used,
     * appropriate for posting on the class website.
     */
    public String algorithmDescription() {
    	return "A kNN implementation with cross-validation.";
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

	Classifier c = new kNN(d);

	d.printTestPredictions(c, filestem);
    }
}
