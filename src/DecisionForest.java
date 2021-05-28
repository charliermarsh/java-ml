/**
 * This is a decision tree class used in a decision forest implementation.  The
 * constructor takes the training set and the attributes to use for building
 * the tree.  Basic algorithm based on R\&N 18.3: Page 702.
 *
 * http://csmr.ca.sandia.gov/~wpk/pubs/publications/pami06.pdf
 */
import java.io.*;
import java.util.Random;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Collections;
public class DecisionForest implements Classifier{

    Random random;
    DecisionTree[] forest;

    private ArrayList<Integer> pickRandomNumber(int number) {
        ArrayList<Integer> result = new ArrayList<Integer>(number);
        for (int i = 0; i < number; i++) { result.add(i); }
        return result;
    }

    private HashSet<Integer> randomizeTreeAttributes(int number, ArrayList<Integer> attributes){
        HashSet<Integer> treeAttributes = new HashSet<Integer>(number);
        Collections.shuffle(attributes);
        for (int i = 0; i < number; i++) {
            treeAttributes.add(attributes.get(i));
        } 
    return treeAttributes;
    }

    private ArrayList<Integer> randomizeTreeExamples(int number, ArrayList<Integer> examples){
        ArrayList<Integer> treeExamples = new ArrayList<Integer>(number);
        Collections.shuffle(examples);
        for (int i = 0; i < number; i++) {
            treeExamples.add(examples.get(i));
        } 
    return treeExamples;
    }
    public DecisionForest(DataSet data, int forestSize) {
        random = new Random();

        forest = new DecisionTree[forestSize];


        
        /*
         * This is basically a nonsensical way of choosing attributes/examples
         * to train each tree on.  Can't really find much on optimal values.
         * Want to make sure the tree works before tweaking, though.
         *
         * Picks a random number of attributes/examples to train each tree.
         */
        ArrayList<Integer> attributes = pickRandomNumber(data.numAttrs);
        ArrayList<Integer> examples = pickRandomNumber(data.numTrainExs);


        //Train each tree by choosing a subset of features. Actually just using
        //every feature in this case.
        int numFeatures = data.numAttrs;
        
        //Train each tree on a sample of 2/3 of the examples
        int numTrain = 2 * data.numTrainExs / 3;

        for (int cTree = 0; cTree < forestSize; cTree++) {

            /* Need to decide how to select number of features*/
            //int numFeatures = random.nextInt(data.numAttrs - 1) + 1;
            //int numTrain = random.nextInt(data.numTrainExs);
            //Randomize the list

            HashSet<Integer> treeAttributes = randomizeTreeAttributes(numFeatures, attributes);
            ArrayList<Integer> treeExamples = randomizeTreeExamples(numTrain, examples);

            //System.out.println(numFeatures + ":" + numTrain);
            forest[cTree] = new DecisionTree(data, treeAttributes,
                                    treeExamples, true);
            //forest[cTree] = new DecisionTree(data, treeAttributes);
        }
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
