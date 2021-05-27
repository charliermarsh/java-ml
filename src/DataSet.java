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
		fileInput.read(this, filestem);
	}

	/**
	 * This method prints out the predictions of classifier <tt>c</tt> on each of
	 * the test examples in the format required for submission. The result is sent
	 * to the given <tt>PrintStream</tt>.
	 **/
	public void printTestPredictions(Classifier c, PrintStream out) {
		FileOutput fileOutput = new FileOutput(out);
		fileOutput.write(this, c);
	}

	/**
	 * This method prints out the predictions of classifier <tt>c</tt> on each of
	 * the test examples in the format required for submission. The result is
	 * printed to the file <tt>filestem.testout</tt>.
	 **/
	public void printTestPredictions(Classifier c, String filestem) throws FileNotFoundException {
		FileOutput fileOutput = new FileOutput();
		fileOutput.write(this, c, filestem);
	}

	/**
	 * @return values (type: int[][]): For numerical attributes, save all values
	 *         shown in train and test examples in ascending order. If the a-th
	 *         attribute is numeric, <tt>values[a][v]</tt> is the v-th value when
	 *         the values of the a-th attribute in <tt>trainEx</tt> and
	 *         <tt>testEx</tt> are sorted in ascending order. Otherwise, array
	 *         corresponding to <tt>values[a]</tt> is not assigned.
	 **/
	protected int[][] getContinuousValues() {
		int[][] values = new int[numAttrs][];
		for (int attributeNum = 0; attributeNum < numAttrs; attributeNum++) {
			if (!isNumericAttribute(attributeNum))
				continue;

			TreeSet<Integer> valueSet = new TreeSet<Integer>();

			addValues(attributeNum, valueSet, trainEx);
			addValues(attributeNum, valueSet, testEx);

			addValues(attributeNum, values, valueSet);
		}
		return values;
	}

	private void addValues(int attributeNum, TreeSet<Integer> destination, int[][] source) {
		for (int i = 0; i < source.length; i++) {
			destination.add( (Integer)source[i][attributeNum] );
		}
	}

	private void addValues(int attributeNum, int[][] destination, TreeSet<Integer> source) {
		destination[attributeNum] = new int[source.size()];
		Iterator<Integer> it = source.iterator();
		int i = 0;
		while (it.hasNext()) {
			destination[attributeNum][i++] = (it.next()).intValue();
		}
	}

	public void setAttributeNumeric(int attributeNum) {
		attrVals[attributeNum] = null;
	}

	public boolean isNumericAttribute(int attributeNum) {
		return attrVals[attributeNum] == null;
	}

}