import java.io.*;

/**
 * This is a subclass of <tt>DataSet</tt> representing a dataset all
 * of whose attributes are discrete.  In addition to all of the fields
 * and methods inherited from <tt>DataSet</tt>, the class includes a
 * constructor that reads in data from data files and converts all
 * numeric attributes to discrete.
 **/
public class DiscreteDataSet extends DataSet {

    /** This constructor reads in data from the files
     * <tt>filestem.names</tt>, <tt>filestem.train</tt> and
     * <tt>filestem.test</tt>, converts all attributes to discrete
     * format, and sets up all of the public fields.
     **/
    public  DiscreteDataSet(String filestem)
	throws FileNotFoundException, IOException {
	super(filestem);

	int[][] cont_vals = getContVals();

	for (int j = 0; j < numAttrs; j++) {
	    if (attrVals[j] != null)
		continue;
	    attrVals[j] = new String[cont_vals[j].length];
	    for (int k = 0; k < cont_vals[j].length; k++) {
		attrVals[j][k] = Integer.toString(cont_vals[j][k]);
	    }
	    for (int traintest = 0; traintest < 2; traintest++) {
		int[][] exs = (traintest == 1 ? trainEx : testEx);
		for (int i = 0; i < exs.length; i++) {
		    int k = 0;
		    while(exs[i][j] != cont_vals[j][k])
			k++;
		    exs[i][j] = k;
		}
	    }
	}
    }

    /** This constructor creates an empty discrete dataset. */
    public DiscreteDataSet() {
	super();
    }

}
