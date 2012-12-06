import java.io.*;

/**
 * This is a subclass of <tt>NumericDataSet</tt> (which, in turn, is a
 * subclass of <tt>DataSet</tt>) representing a dataset all of whose
 * attributes are binary, i.e., numeric with only the two values 0 and
 * 1.  In addition to all of the fields and methods inherited from
 * <tt>DataSet</tt>, the class includes a constructor that reads in
 * data from data files and converts all attributes to binary.
 **/
public class BinaryDataSet extends NumericDataSet {

    /** This constructor reads in data from the files
     * <tt>filestem.names</tt>, <tt>filestem.train</tt> and
     * <tt>filestem.test</tt>, converts all attributes to binary
     * format, and sets up all of the public fields.
     **/
    public BinaryDataSet(String filestem)
	throws FileNotFoundException, IOException {
	super(filestem);

	int new_numAttrs = 0;

	int[][] cont_vals = getContVals();

	for (int a = 0; a < numAttrs; a++)
	    new_numAttrs += cont_vals[a].length - 1;

	for (int traintest = 0; traintest < 2; traintest++) {
	    int[][] exs = (traintest == 1 ? trainEx : testEx);
	    for (int i = 0; i < exs.length; i++) {
		int[] new_ex = new int[new_numAttrs];
		for (int a = 0, na = 0; a < numAttrs; a++) {
		    for(int j = 1; j < cont_vals[a].length; j++)
			new_ex[na++] = (exs[i][a] >= cont_vals[a][j]
					? 1 : 0);
		}
		exs[i] = new_ex;
	    }
	}

	String[] new_attrName = new String[new_numAttrs];
	for (int a = 0, na = 0; a < numAttrs; a++) {
	    if (cont_vals[a].length == 2
		&& cont_vals[a][0] == 0
		&& cont_vals[a][1] == 1)
		new_attrName[na++] = attrName[a];
	    else
		for(int j = 1; j < cont_vals[a].length; j++)
		    new_attrName[na++] = attrName[a]
			+ ">=" + cont_vals[a][j];
	}

	attrName = new_attrName;
	numAttrs = new_numAttrs;

	attrVals = new String[numAttrs][];
    }

    /** This constructor creates an empty binary dataset. */
    public BinaryDataSet() {
	super();
    }

}
