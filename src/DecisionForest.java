/**
 * This is a decision tree class used in a decision forest implementation.  The
 * constructor takes the training set and the attributes to use for building
 * the tree.  Basic algorithm based on R\&N 18.3: Page 702.
 */
import java.io.*;
import java.util.Random;
import java.util.Vector;
public class DecisionForest implements Classifier{

    Random random;
    DecisionTree[] forest;

    public DecisionForest(BinaryDataSet data, int forestSize) {
        random = new Random();

        forest = new DecisionTree[forestSize];
        
        int[] attributes = new int[data.numAttrs];
        for (int i = 0; i < data.numAttrs; i++) { attributes[i] = i; }

        for (int cTree = 0; cTree < forestSize; cTree++) {

            /* Need to decide how to select number of features*/
            int numFeatures = random.nextInt(data.numAttrs / 2) + 1;
            Vector<Integer> treeAttributes = new Vector<Integer>(numFeatures);
        
            for (int i = 1; i <= numFeatures; i++) {
                //Basically do a partial knuth shuffle to get random attributes
                int index = random.nextInt(data.numAttrs - i);
                int value = attributes[index];
                attributes[index] = attributes[data.numAttrs - i];
                attributes[data.numAttrs - i] = value;

                treeAttributes.add(value);
                //System.out.print(value + " ");
            }
                //System.out.println();
                forest[cTree] = new DecisionTree(data, treeAttributes);
        }
    }

    /*
     * Takes the majority vote of the decision trees
     */
    public int predict(int[] ex) {
        int[] count = new int[2];
        for (DecisionTree tree: forest)
            count[tree.predict(ex)]++;
        return (count[1] > count[0] ? 1 : 0);
    }

    /** This method should return a very brief but understandable
     * description of the learning algorithm that is being used,
     * appropriate for posting on the class website.
     */
    public String algorithmDescription() {
        return "Basic decision forest - uses our DecisionTree";
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

        if (argv.length < 2) {
            System.err.println("argument: filestem forestSize");
            return;
        }

        String filestem = argv[0];

        BinaryDataSet d = new BinaryDataSet(filestem);

        Classifier c = new DecisionForest(d,Integer.parseInt(argv[1]));

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
