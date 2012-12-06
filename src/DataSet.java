import java.util.*;
import java.io.*;

/**
 * This class represents a dataset, including names for the classes,
 * all attributes and their values.  The class also includes a
 * constructor that can read the dataset from data files, as well as a
 * method for printing the predictions of a classifier on each of the
 * test examples in the format required for submission.
 **/
public class DataSet {

    /** number of training examples **/
    public int numTrainExs;

    /** number of test examples **/
    public int numTestExs;

    /** an array of training examples, each of which is itself an
     * array of integer values so that <tt>trainEx[i][a]</tt> is the
     * value of attribute <tt>a</tt> on example <tt>i</tt> **/
    public int trainEx[][];

    /** an array of labels for the training examples **/
    public int trainLabel[];

    /** an array of test examples, each one an array of integer values **/
    public int testEx[][];

    /** number of attributes **/
    public int numAttrs;

    /** the names of the attributes **/
    public String attrName[];

    /** an array of names for the attribute values:
     * for discrete attribute <tt>a</tt>, <tt>attrVals[a][j]</tt> is
     * the name of the <tt>j</tt>-th value; if <tt>a</tt> is a numeric
     * attribute then <tt>attrVals[a]</tt> is <tt>null</tt> **/
    public String attrVals[][];

    /** names of the two classes **/
    public String className[] = null;

    /** This constructor constructs an empty dataset with no training
     * examples, no test examples, no attributes, and two classes with
     * default names.
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

    /** This constructor reads in data from the files
     * <tt>filestem.names</tt>, <tt>filestem.train</tt> and
     * <tt>filestem.test</tt>, and then sets up all of the public
     * fields.  See assignment instructions for information on the
     * required format of these files. 
     **/
    public DataSet(String filestem)
	throws FileNotFoundException, IOException {
	String[] words = null;

	// read .names file 

	open_file(filestem + ".names");
	ArrayList<String[]> attr_list = new ArrayList<String[]>();

	String line;
	while((line = read_line()) != null) {
	    line = line.trim( );
	    words = line.split("\\s+");
	    if (line.equals(""))
		continue;

	    if (className == null) {
		if (words.length != 2) {
		    String err = "expected two class names at line "
			+ line_count + " in file " + filename;
		    System.err.println(err);
		    throw new RuntimeException(err);
		}
		className = words;
	    } else {
		if (words.length <= 1) {
		    String err = "expected attribute description at line "
			+ line_count + " in file " + filename;
		    System.err.println(err);
		    throw new RuntimeException(err);
		}
		attr_list.add(words);
		numAttrs++;
	    }
	}

	in.close();

	attrName = new String[numAttrs];
	attrVals = new String[numAttrs][];

	for (int i = 0; i < numAttrs; i++) {
	    words = attr_list.get(i);
	    attrName[i] = words[0];
	    if (words[1].equals("numeric")) {
		attrVals[i] = null;
	    } else {
		attrVals[i] = new String[words.length - 1];
		for (int j = 1; j < words.length; j++) {
		    attrVals[i][j-1] = words[j];
		}
	    }
	}

	// read data files

	for(int traintest = 0; traintest < 2; traintest++) {
	    ArrayList<int[]> ex_list = new ArrayList<int[]>();
	    ArrayList<Integer> lab_list = new ArrayList<Integer>();

	    if (traintest == 1)
		open_file(filestem + ".train");
	    else
		try {
		    open_file(filestem + ".test");
		} catch (FileNotFoundException e) {
		    System.err.print("Continuing without test file...\n");
		    numTestExs = 0;
		    testEx = new int[0][];
		    continue;
		}

	    while((line = read_line()) != null) {
		line = line.trim( );
		if (line.equals(""))
		    continue;

		words = line.split("\\s+");
		if (words.length != numAttrs + traintest) {
		    String err = "wrong number of tokens at line "
			+ line_count + " in file " + filename;
		    System.err.println(err);
		    throw new RuntimeException(err);
		}
		    
		int ex[] = new int[numAttrs];
		for (int i = 0; i < numAttrs; i++) {
		    if (attrVals[i] == null) {
			try {
			    ex[i] = Integer.parseInt(words[i]);
			} catch (NumberFormatException e) {
			    System.err.println("Expected integer in field "
					       +(i+1)+" at line "+line_count+
					       " in file "+filename);
			    throw e;
			}
		    } else {
			int j;
			for (j = 0;
			     j < attrVals[i].length
				 && !attrVals[i][j].equals(words[i]);
			     j++);
			if (j >= attrVals[i].length) {
			    String err = "bad attribute value in field "
				+(i+1)+" at line "+line_count+" in file "+filename;
			    System.err.println(err);
			    throw new RuntimeException(err);
			}
			ex[i] = j;
		    }
		}
		ex_list.add(ex);
		if (traintest == 1) {
		    int lab;
		    if (words[numAttrs].equals(className[0])) {
			lab = 0;
		    } else if (words[numAttrs].equals(className[1])) {
			lab = 1;
		    } else {
			String err = "unrecognized label at line "+line_count+
			    " in file "+filename;
			System.err.println(err);
			throw new RuntimeException(err);
		    }
		    lab_list.add(new Integer(lab));
		}
	    }

	    if (traintest == 0) {
		numTestExs = ex_list.size();
		testEx = new int[0][];
		testEx = (int[][]) ex_list.toArray(testEx);
	    } else {
		numTrainExs = ex_list.size();
		trainEx = new int[0][];
		trainEx = (int[][]) ex_list.toArray(trainEx);
		trainLabel = new int[numTrainExs];
		for (int i = 0; i < numTrainExs; i++) {
		    trainLabel[i] = (lab_list.get(i)).intValue();
		}
	    }
	    in.close();
	}
	in = null;
	filename = null;
    }

    /** This method prints out the predictions of classifier
     * <tt>c</tt> on each of the test examples in the format required
     * for submission.  The result is sent to the given
     * <tt>PrintStream</tt>.
     **/
    public void printTestPredictions(Classifier c,
				     PrintStream out) {
	out.println(c.author());
	out.println(".");
	out.println(c.algorithmDescription());
	out.println(".");
	for(int i = 0; i < numTestExs; i++) {
	    out.println(className[c.predict(testEx[i])]);
	}
    }

    /** This method prints out the predictions of classifier
     * <tt>c</tt> on each of the test examples in the format required
     * for submission.  The result is printed to the file
     * <tt>filestem.testout</tt>.
     **/
    public void printTestPredictions(Classifier c,
				     String filestem)
    throws FileNotFoundException {
	PrintStream out;

	try {
	    out = new PrintStream(new BufferedOutputStream(new
		FileOutputStream(filestem + ".testout")));
	} catch (FileNotFoundException e) {
	    System.err.println("Cannot open file " + filestem + ".testout");
	    throw e;
	}
	printTestPredictions(c, out);

	out.close();
    }

    /*********************** private ********************************/

    private String filename;
    private int line_count;
    private BufferedReader in;

    private void open_file(String filename) throws FileNotFoundException {
	BufferedReader in;
	
	this.filename = filename;
	this.line_count = 0;

	try {
	    in = new BufferedReader(new FileReader(filename));
	} catch (FileNotFoundException e) {
	    System.err.print("File "+filename+" not found.\n");
	    throw e;
	}
	this.in = in;
    }

    private String read_line() throws IOException {
	String line;

	line_count++;

	try {
	    line = in.readLine();
	}
	catch (IOException e) {
	    System.err.println("Error reading line "+line_count+" in file "+filename);
	    throw e;
	}
	return line;
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
	    while(it.hasNext()) {
		vals[a][i++] = (it.next()).intValue();
	    }
	}
	return vals;
    }

}    
