import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;

public abstract class TemplateMethod implements Classifier {
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
    
    protected abstract void makeTree(DataSet data, HashSet<Integer> treeAttributes, ArrayList<Integer> treeExamples, int cTree);
	
	public TemplateMethod(DataSet data, int forestSize) {
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
            makeTree(data, treeAttributes, treeExamples, cTree);
            //forest[cTree] = new DecisionTree(data, treeAttributes);

        }    
            
	}

}