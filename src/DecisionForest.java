/**
 * This is a decision tree class used in a decision forest implementation.  The
 * constructor takes the training set and the attributes to use for building
 * the tree.  Basic algorithm based on R\&N 18.3: Page 702.
 *
 * http://csmr.ca.sandia.gov/~wpk/pubs/publications/pami06.pdf
 */
import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
public class DecisionForest extends TemplateMethod{

	@Override
    protected void makeTree(DataSet data, HashSet<Integer> treeAttributes, ArrayList<Integer> treeExamples, int cTree) {
    	forest[cTree] = new DecisionTree(data, treeAttributes, treeExamples, true);
		
    }


	public DecisionForest(DataSet data, int forestSize) {
		super(data, forestSize);
	}
    /*
     * Takes the majority vote of the decision trees
     */
    public int predict(int[] ex) {
        int[] count = new int[2];
        for (DecisionTree tree : forest)
            count[tree.predict(ex)]++;
        return (count[1] > count[0] ? 1 : 0);
    }

    /** This method should return a very brief but understandable
     * description of the learning algorithm that is being used,
     * appropriate for posting on the class website.
     */
    public String algorithmDescription() {
        return "Random forest - combines bagging with unpruned random decision trees.  Tree chooses best among subset of attributes to split on at each node.";
    }

    /** This method should return the "author" of this program as you
     * would like it to appear on the class website.  You can use your
     * real name, or a pseudonym, or a name that identifies your
     * group.
     */
    public String author() {
        return "crm & dmrd";
    }

    /*
     * Simple main for testing.
     *
     */
    public static void main(String argv[])
        throws FileNotFoundException, IOException {

        if (argv.length < 2) {
            System.err.println("argument: filestem forestSize");
            return;
        }

        String filestem = argv[0];

        /*
         * Create a cross validation set - just takes the last crossSize
         * elements of the set as a cross set.
         */
        DiscreteDataSet d = new DiscreteDataSet(filestem);

        System.out.println("Training classifier on " + d.numTrainExs
                + " examples");

        Classifier c = new DecisionForest(d,Integer.parseInt(argv[1]));

        System.out.println("Running on test set...");
        d.printTestPredictions(c, filestem);
    }



}
