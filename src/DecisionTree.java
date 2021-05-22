/**
 * This is a decision tree class used in a decision forest implementation.  The
 * constructor takes the training set and the attributes to use for building
 * the tree.  Basic algorithm based on R\&N 18.3: Page 702.
 */
import java.io.*;
import java.util.Random;
import java.util.ArrayList;
import java.util.HashSet;
public class DecisionTree implements Classifier{

    private Node treeRoot;

    private Random random;

    /*
     * Randomize tree (for random forest)?  If true, chooses some subset of
     * attributes to choose best from at each node
     */
    private boolean randomize;



    public void randomizedTree(boolean rand) {
    	random = new Random();
    	this.randomize = rand;
    }
    
    public HashSet<Integer> initializeAttributes(DataSet data) {
    	HashSet<Integer> attributes = new HashSet<Integer>(data.numAttrs);
    	for (int i = 0; i < data.numAttrs; i++) { attributes.add(i); }
		return attributes;
    }
 
    public ArrayList<Integer> initializeExamples(DataSet data) {
        ArrayList<Integer> examples = new ArrayList<Integer>(data.numTrainExs);
        for (int i = 0; i < data.numTrainExs; i++) { examples.add(i); }
		return examples;
    }
    
    /*Just takes dataset - uses all attributes in training*/
    public DecisionTree(DataSet data, boolean rand) {
    	randomizedTree(rand); 
    	treeRoot = new Node(data, initializeAttributes(data), initializeExamples(data), randomize, random);
    }

    /*Takes the dataset and attributes to use in training*/
    public DecisionTree(DataSet data, HashSet<Integer> attributes, boolean rand) {
    	randomizedTree(rand);
        treeRoot = new Node(data, attributes, initializeExamples(data), randomize, random);
    }

    /*Take both attributes and examples to use for training*/
    public DecisionTree(DataSet data, HashSet<Integer> attributes,
            ArrayList<Integer> examples, boolean rand) {
    	randomizedTree(rand);
        treeRoot = new Node(data, attributes, examples, randomize, random);
    }

    /**  
     * Walks down the generated tree to return a label for the example.
     */
    public int predict(int[] ex) {
        Node current = treeRoot;
        int depth = 0;
        while (current.attribute != -1) {
            current = current.children[ex[current.attribute]];
        }
        //System.out.println(depth);
        //System.out.println(current.attribute + "\t" + current.label);
        return current.label;
    }

    /** This method should return a very brief but understandable
     * description of the learning algorithm that is being used,
     * appropriate for posting on the class website.
     */
    public String algorithmDescription() {
        return "Basic decision tree for use with random forests";
    }

    /** This method should return the "author" of this program as you
     * would like it to appear on the class website.  You can use your
     * real name, or a pseudonym, or a name that identifies your
     * group.
     */
    public String author() {
        return "dmrd";
    }

    /*
     * Simple main for testing.
     */
    public static void main(String argv[])
        throws FileNotFoundException, IOException {

        if (argv.length < 1) {
            System.err.println("argument: filestem");
            return;
        }

        String filestem = argv[0];

        /*
         * Create a cross validation set - just takes the last crossSize
         * elements of the set as a cross set.
         */
        DiscreteDataSet d = new DiscreteDataSet(filestem);

        /*
         * Do the Knuth Shuffle!  It sounds like more fun than it is!
         */
        //Set seed to constant to get the same result multiple times
        Random random = new Random();
        for (int i = 0; i < d.numTrainExs; i++) {
            int swap = random.nextInt(d.numTrainExs - i);
            int[] tempEx = d.trainEx[swap];
            d.trainEx[swap] = d.trainEx[d.numTrainExs - i - 1];
            d.trainEx[d.numTrainExs - i - 1] = tempEx;
            int tempLabel = d.trainLabel[swap];
            d.trainLabel[swap] = d.trainLabel[d.numTrainExs - i - 1];
            d.trainLabel[d.numTrainExs - i - 1] = tempLabel;
        }

        //What proportion of the dataset to use for testing
        int crossSize = d.numTrainExs/8;

        int[][] crossEx = new int[crossSize][];
        int[] crossLabel = new int[crossSize];

        int[][] dEx = new int[d.numTrainExs - crossSize][];
        int[] dLabel = new int[d.numTrainExs - crossSize];

        for (int i = 0; i < d.numTrainExs - crossSize; i++) {
            dEx[i] = d.trainEx[i];
            dLabel[i] = d.trainLabel[i];
        }

        for (int i = 0; i < crossSize; i++) {
            crossEx[i] = d.trainEx[d.numTrainExs - i - 1];
            crossLabel[i] = d.trainLabel[d.numTrainExs - i - 1];
        }

        //Modify original dataset
        d.numTrainExs = dEx.length;
        d.trainEx = dEx;
        d.trainLabel = dLabel;
        System.out.println("Training classifier on " + d.numTrainExs
                + " examples");

        Classifier c = new DecisionTree(d, false);

        System.out.println("Testing classifier on " + crossEx.length
                + " examples");
        int correct = 0;
        for (int ex = 0; ex < crossEx.length; ex++) {
            if (c.predict(crossEx[ex]) == crossLabel[ex])
                correct++;
        }
        System.out.println("Performance on cross set: "
                + (100*correct / crossEx.length) + "%");
    }
}
