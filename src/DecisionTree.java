/**
 * This is a decision tree class used in a decision forest implementation.  The
 * constructor takes the training set and the attributes to use for building
 * the tree.  Basic algorithm based on R\&N 18.3: Page 702.
 */
import java.io.*;
import java.util.Random;
import java.util.Vector;
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
         * What is the label for this node?  Only set if attribute is -1
         */
        public int label;
        /* Left branch of node */
        public Node zeroBranch;
        /* Right branch of node - */
        public Node oneBranch;

        /*
         * Accepts the dataset then an array of indices for remaining
         * attributes and examples.  So each integer in examples is just the
         * index of a training example in data.trainEx.  Avoid making too many
         * copies of the examples this way
         */
        Node(BinaryDataSet data,  Vector<Integer> attributes,
                Vector<Integer> examples, int attrRemaining) {
            Vector<Integer> zeroExamples = new Vector<Integer>();
            Vector<Integer> oneExamples = new Vector<Integer>();

            this.label = -1;

            if (examples.size() == 0) {
                this.attribute = -1;
                return; //Parent must set the label on this node
            }

            int majority = 0;
            int count[] = new int[2];

            for (int i = 0; i < examples.size(); i++)
                count[data.trainLabel[examples.get(i)]]++;

            majority = (count[1] > count[0] ? 1 : 0);

            /* They all have the same label or there are no more attributes to
             * split on*/
            if (count[majority] == examples.size() || attrRemaining == 0) {
                this.attribute = -1;
                this.label = majority;
                return;
            }

            /*Generate attribute to split on*/
            int splitAttrIndex = random.nextInt(attrRemaining);
            this.attribute = attributes.get(splitAttrIndex);

            /*
             * Move this attribute to the end - we avoid making copies of the
             * attribute array this way.
             */
            int swapAttr = attributes.get(attrRemaining - 1);
            attributes.set(attrRemaining - 1, this.attribute);
            attributes.set(splitAttrIndex, swapAttr);

            /*
             * Split examples based on the randomly chosen attribute
             */
            for (int i = 0; i < examples.size(); i++) {
                if (data.trainEx[i][this.attribute] == 0) {
                    zeroExamples.add(i);
                } else {
                    oneExamples.add(i);
                }
            }

            /*
             *System.out.println(majority + "\t" + zeroExamples.size() + "\t"
             *        + oneExamples.size() + "\t" + attrRemaining + "\t"
             *        + this.attribute + "\t" + this.label);
             */

            /*Create children trees*/
            zeroBranch = new Node(data, attributes,
                    zeroExamples, attrRemaining - 1);
            oneBranch = new Node(data, attributes,
                    oneExamples, attrRemaining - 1);
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
        }
    }

    /*Just takes dataset - uses all attributes in training*/
    public DecisionTree(BinaryDataSet data) {
        random = new Random();

        Vector<Integer> attributes = new Vector<Integer>(data.numAttrs);
        Vector<Integer> examples = new Vector<Integer>(data.numTrainExs);

        /*Initialize example and attribute vectors*/
        for (int i = 0; i < data.numAttrs; i++) { attributes.add(i); }
        for (int i = 0; i < data.numTrainExs; i++) { examples.add(i); }

        treeRoot = new Node(data, attributes, examples, attributes.size());
    }

    /*Takes the dataset and attributes to use in training*/
    public DecisionTree(BinaryDataSet data, Vector<Integer> attributes) {
        random = new Random();

        /*Initialize example vector to include all examples*/
        Vector<Integer> examples = new Vector<Integer>(data.numTrainExs);
        for (int i = 0; i < data.numTrainExs; i++) { examples.add(i); }

        treeRoot = new Node(data, attributes, examples, attributes.size());
    }


    /** A method for predicting the label of a given example <tt>ex</tt>
     * represented, as in the rest of the code, as an array of values
     * for each of the attributes.  The method should return a
     * prediction, i.e., 0 or 1.
     */
    public int predict(int[] ex) {
        Node current = treeRoot;
        while (current.attribute != -1) {
            //System.out.println(current.attribute + "\t" + current.label);
            if (ex[current.attribute] == 0) {
                current = current.zeroBranch;
            } else {
                current = current.oneBranch;
            }
        }
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

        BinaryDataSet d = new BinaryDataSet(filestem);

        Classifier c = new DecisionTree(d);


        /*
         * Give some estimate of performance - using training set is bad form,
         * but it's all we have for now
         */
        int correct = 0;
        for (int i = 0; i < d.numTrainExs; i++) {
            if (c.predict(d.trainEx[i]) == d.trainLabel[i])
                correct++;
        }

        System.out.println((100*correct/d.numTrainExs) + "%");

        //d.printTestPredictions(c, System.out);
        //d.printTestPredictions(c, filestem);
    }

}
