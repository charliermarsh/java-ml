/**
 * This is a decision tree class used in a decision forest implementation.  The
 * constructor takes the training set and the attributes to use for building
 * the tree.  Basic algorithm based on R\&N 18.3: Page 702.
 */
import java.io.*;
import java.util.Random;
import java.util.Vector;
import java.util.HashSet;
public class DecisionTree implements Classifier{

    Node treeRoot;
    Random random;

    /*
     *  Inner class to represent tree structure - Splits on binary features
     */
    private class Node {
        /*
         * What attribute does this branch split on? -1 indicates nothing
         */
        public int attribute;
        /*
         * What is the label for this node?
         */
        public int label;
        /* Left branch of node */
        public Node zeroBranch;
        /* Right branch of node - */
        public Node oneBranch;

        /*
         * Entropy of boolean variable - n negative examples, p positive
         * examples
         */
        double entropy(double n, double p) {
            if (n == 0 || p == 0)
                return 0.0;
            return -1.0
                * ( ((n/(n+p)) * Math.log(n/(n+p)))
                      + ( (p/(n+p)) * Math.log(p/(n+p)))) / Math.log(2);
        }

        /* 
         * Pick the most important attribute - want to get the best one to
         * split on - see R&N 18.3.4 on page 703.  Trying to minimize
         * Remainder(A) since B(.) will remain the same for every
         * attribute.
         *
         * Calculates max gain - for a better explanation, see
         * http://dms.irb.hr/tutorial/tut_dtrees.php
         * http://decisiontrees.net/decision-trees-tutorial/tutorial-4-id3/
         */
        int chooseAttribute(BinaryDataSet data,  HashSet<Integer> attributes,
                Vector<Integer> examples) {
            int bestAttr = -1;
            double bestGain = 0;
            for (int attr : attributes) {
                /*
                 * [value][label] : so [0][0] + [0][1] would be the number of
                 * examples with a value of 0 for the attributes.
                 * [0][1] + [1][1] would be number of examples with label 1 in
                 * examples.  Use this to calculate entropy and gain.
                 */
                double[][] count = new double[2][2];
                for (int ex : examples) {
                    //Yay array index beauty...
                    count[data.trainEx[ex][attr]][data.trainLabel[ex]]++;
                }
                
                //Not needed, but using until I am sure my code works
                //(recalculates current entropy every time)
                //double gain = entropy(count[0][0] + count[1][0], count[0][1]
                        //+ count[1][1]);
                double gain = 1.0;
                double zeroEntropy = (count[0][0] + count[0][1])
                                        / examples.size();
                zeroEntropy *= entropy(count[0][0], count[0][1]);

                double oneEntropy = (count[1][0] + count[1][1])
                                        / examples.size();
                oneEntropy *= entropy(count[1][0], count[1][1]);

                gain -= zeroEntropy + oneEntropy;
                if (gain >= bestGain) {
                    bestAttr = attr;
                    bestGain = gain;
                }
                //System.out.println(gain);
            }
            //System.out.println(bestGain + "\n\n\n");
            return bestAttr;
        }

        /*
         * Accepts the dataset then an array of indices for remaining examples.
         * So each integer in examples is just the index of a training example
         * in data.trainEx.  Avoid making too many copies of the examples this
         * way.  attributes is a set of all remaining attributes to split on.
         * This is modified and passed to children by each node.
         */
        Node(BinaryDataSet data,  HashSet<Integer> attributes,
                Vector<Integer> examples) {

            Vector<Integer> zeroExamples = new Vector<Integer>();
            Vector<Integer> oneExamples = new Vector<Integer>();

            this.label = -1;

            if (examples.size() == 0) {
                this.attribute = -1;
                this.label = 0; //To avoid crashes, although parent must set
                return;         //      the label on this node
            }

            /*
             * Calculates majority class
             */
            int majority = 0;
            int count[] = new int[2];

            for (int ex : examples) {
                count[data.trainLabel[ex]]++;
            }

            majority = (count[1] > count[0] ? 1 : 0);

            //System.out.println(attributes.size() + "\t" + examples.size()
            //        + "\t" + count[0] + ":" + count[1]);

            /*
             * They all have the same label or there are no more attributes to
             * split on
             */
            if (count[majority] == examples.size() || attributes.size() == 0) {
                //System.out.println(count[0] + ":" + count[1]);
                this.attribute = -1;
                this.label = majority;
                return;
            }


            this.attribute = chooseAttribute(data, attributes, examples);

            //Remove the attribute so it cannot be used again in child branches
            //Add it back in before returning
            attributes.remove(this.attribute);

            /*
             * Split examples based on the chosen attribute
             */
            for (int i = 0; i < examples.size(); i++) {
                if (data.trainEx[i][this.attribute] == 0) {
                    zeroExamples.add(i);
                } else {
                    oneExamples.add(i);
                }
            }

            /*Create children trees*/
            zeroBranch = new Node(data, attributes, zeroExamples);
            oneBranch = new Node(data, attributes, oneExamples);

            /*
             * Need to set child label if they don't have any examples to train
             * on
             */
            if (zeroExamples.size() == 0) {
                zeroBranch.label = majority;
            }
            if (oneExamples.size() == 0) {
                oneBranch.label = majority;
            }

            attributes.add(this.attribute);
        }
    }

    /*Just takes dataset - uses all attributes in training*/
    public DecisionTree(BinaryDataSet data) {
        random = new Random();

        HashSet<Integer> attributes = new HashSet<Integer>(data.numAttrs);
        Vector<Integer> examples = new Vector<Integer>(data.numTrainExs);

        /*Initialize example and attribute vectors*/
        for (int i = 0; i < data.numAttrs; i++) { attributes.add(i); }
        for (int i = 0; i < data.numTrainExs; i++) { examples.add(i); }

        treeRoot = new Node(data, attributes, examples);
    }

    /*Takes the dataset and attributes to use in training*/
    public DecisionTree(BinaryDataSet data, HashSet<Integer> attributes) {
        random = new Random();

        /*Initialize example vector to include all examples*/
        Vector<Integer> examples = new Vector<Integer>(data.numTrainExs);
        for (int i = 0; i < data.numTrainExs; i++) { examples.add(i); }

        treeRoot = new Node(data, attributes, examples);
    }

    /*Take both attributes and examples to use for training*/
    public DecisionTree(BinaryDataSet data, HashSet<Integer> attributes,
            Vector<Integer> examples) {
        random = new Random();
        treeRoot = new Node(data, attributes, examples);
    }

    /**  
     * Walks down the generated tree to return a label for the example.
     */
    public int predict(int[] ex) {
        Node current = treeRoot;
        int depth = 0;
        while (current.attribute != -1) {
            depth++;
            //System.out.println(current.attribute + "\t" + current.label);
            if (ex[current.attribute] == 0) {
                current = current.zeroBranch;
            } else {
                current = current.oneBranch;
            }
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
     *
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
        BinaryDataSet d = new BinaryDataSet(filestem);

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

        Classifier c = new DecisionTree(d);

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
