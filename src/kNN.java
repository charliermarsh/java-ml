import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;

public class kNN implements Classifier {
	
	private int[][] labelledData;
	private int labelIndex;
	private int k = 7;

	/* Computes the squared distance between two numeric
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
	}
	
	private class columnComparator implements Comparator {
	    private int[] columnToSortOn;
	    
	    //contructor to set the column to sort on.
	    private columnComparator(int[] columnToSortOn) {
	      this.columnToSortOn = columnToSortOn;
	    }

	    // Implement the abstract method which tells
	    // how to order the two elements in the array.
	    public int compare(Object o1, Object o2) {
	    // cast the object args back to string arrays
	    	int[] a1 = (int[])o1;
	        int[] a2 = (int[])o2;
	        
	        double d1 = dist(a1, this.columnToSortOn);
	        double d2 = dist(a2, this.columnToSortOn);
			if (d1 > d2) return 1;
			if (d1 < d2) return -1;
			return 0;
	    }
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
    	
    	int vote_1 = 0;
    	for (int i = 0; i < this.k; i++)
    		if (this.labelledData[i][this.labelIndex] == 1.0)
    			vote_1++;
    	return (vote_1 > this.k/2)? 1 : 0;
    }

    /** This method should return a very brief but understandable
     * description of the learning algorithm that is being used,
     * appropriate for posting on the class website.
     */
    public String algorithmDescription() {
    	return "A kNN implementation.";
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

	DataSet d = new NumericDataSet(filestem);

	Classifier c = new kNN(d);

	d.printTestPredictions(c, filestem);
    }
}
