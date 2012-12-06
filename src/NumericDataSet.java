import java.io.*;

/**
 * This is a subclass of <tt>DataSet</tt> representing a dataset all
 * of whose attributes are numeric.  In addition to all of the fields
 * and methods inherited from <tt>DataSet</tt>, the class includes a
 * constructor that reads in data from data files and converts all
 * discrete attributes to numeric.
 **/
public class NumericDataSet extends DataSet {

    /** This constructor reads in data from the files
     * <tt>filestem.names</tt>, <tt>filestem.train</tt> and
     * <tt>filestem.test</tt>, converts all attributes to numeric
     * format, and sets up all of the public fields.
     **/
    public NumericDataSet(String filestem)
	throws FileNotFoundException, IOException {
	super(filestem);

	int new_numAttrs = 0;

	for (int a = 0; a < numAttrs; a++)
	    new_numAttrs += (attrVals[a] == null ? 1 : attrVals[a].length);

	for (int traintest = 0; traintest < 2; traintest++) {
	    int[][] exs = (traintest == 1 ? trainEx : testEx);
	    for (int i = 0; i < exs.length; i++) {
		int[] new_ex = new int[new_numAttrs];
		for (int a = 0, na = 0; a < numAttrs; a++) {
		    if (attrVals[a] == null)
			new_ex[na++] = exs[i][a];
		    else
			for(int j = 0; j < attrVals[a].length; j++)
			    new_ex[na++] = (exs[i][a] == j ? 1 : 0);
		}
		exs[i] = new_ex;
	    }
	}

	String[] new_attrName = new String[new_numAttrs];
	for (int a = 0, na = 0; a < numAttrs; a++) {
	    if (attrVals[a] == null)
		new_attrName[na++] = attrName[a];
	    else
		for(int j = 0; j < attrVals[a].length; j++)
		    new_attrName[na++] = attrName[a] + "=" + attrVals[a][j];
	}

	attrName = new_attrName;
	numAttrs = new_numAttrs;

	attrVals = new String[numAttrs][];
    }

    /** This constructor creates an empty numeric dataset. */
    public NumericDataSet() {
	super();
    }

}
