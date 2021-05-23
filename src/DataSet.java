import java.util.*;
import java.io.*;

/**
 * This class represents a dataset, including names for the classes, all
 * attributes and their values. The class also includes a constructor that can
 * read the dataset from data files, as well as a method for printing the
 * predictions of a classifier on each of the test examples in the format
 * required for submission.
 **/
public class DataSet {

	/** number of training examples **/
	public int numTrainExs;

	/** number of test examples **/
	public int numTestExs;

	/**
	 * an array of training examples, each of which is itself an array of integer
	 * values so that <tt>trainEx[i][a]</tt> is the value of attribute <tt>a</tt> on
	 * example <tt>i</tt>
	 **/
	public int trainEx[][];

	/** an array of labels for the training examples **/
	public int trainLabel[];

	/** an array of test examples, each one an array of integer values **/
	public int testEx[][];

	/** number of attributes **/
	public int numAttrs;

	/** the names of the attributes **/
	public String attrName[];

	/**
	 * an array of names for the attribute values: for discrete attribute
	 * <tt>a</tt>, <tt>attrVals[a][j]</tt> is the name of the <tt>j</tt>-th value;
	 * if <tt>a</tt> is a numeric attribute then <tt>attrVals[a]</tt> is
	 * <tt>null</tt>
	 **/
	public String attrVals[][];

	/** names of the two classes **/
	public String className[] = null;

	/**
	 * This constructor constructs an empty dataset with no training examples, no
	 * test examples, no attributes, and two classes with default names.
	 */
	public DataSet() {
		numTrainExs = 0;
		numTestExs = 0;
		trainEx = new int[0][];
		trainLabel = new int[0];
		testEx = new int[0][];
		numAttrs = 0;
		attrName = new String[0];
		attrVals = new String[0][];
		className = new String[2];
		className[0] = "0";
		className[1] = "1";
	}

	/**
	 * This constructor reads in data from the files <tt>filestem.names</tt>,
	 * <tt>filestem.train</tt> and <tt>filestem.test</tt>, and then sets up all of
	 * the public fields. See assignment instructions for information on the
	 * required format of these files.
	 **/
	public DataSet(String filestem) throws FileNotFoundException, IOException {
		FileInput fileInput = new FileInput();
		fileInput.readAttributeInformation(this, filestem);
		fileInput.readExamples(this, filestem);
	}
	
	/**
	 * This method prints out the predictions of classifier <tt>c</tt> on each of
	 * the test examples in the format required for submission. The result is sent
	 * to the given <tt>PrintStream</tt>.
	 **/
	public void printTestPredictions(Classifier c, PrintStream out) {
		out.println(c.author());
		out.println(".");
		out.println(c.algorithmDescription());
		out.println(".");
		for (int i = 0; i < numTestExs; i++) {
			out.println(className[c.predict(testEx[i])]);
		}
	}

	/**
	 * This method prints out the predictions of classifier <tt>c</tt> on each of
	 * the test examples in the format required for submission. The result is
	 * printed to the file <tt>filestem.testout</tt>.
	 **/
	public void printTestPredictions(Classifier c, String filestem) throws FileNotFoundException {
		PrintStream out;

		try {
			out = new PrintStream(new BufferedOutputStream(new FileOutputStream(filestem + ".testout")));
		} catch (FileNotFoundException e) {
			System.err.println("Cannot open file " + filestem + ".testout");
			throw e;
		}
		printTestPredictions(c, out);

		out.close();
	}
	
	protected int[][] getContVals() {
		int[][] vals = new int[numAttrs][];
		for (int a = 0; a < numAttrs; a++) {
			if (attrVals[a] != null)
				continue;

			TreeSet<Integer> t = new TreeSet<Integer>();

			for (int traintest = 0; traintest < 2; traintest++) {
				int[][] exs = (traintest == 1 ? trainEx : testEx);
				for (int i = 0; i < exs.length; i++) {
					t.add(new Integer(exs[i][a]));
				}
			}

			vals[a] = new int[t.size()];

			Iterator<Integer> it = t.iterator();
			int i = 0;
			while (it.hasNext()) {
				vals[a][i++] = (it.next()).intValue();
			}
		}
		return vals;
	}

}