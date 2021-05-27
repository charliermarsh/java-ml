import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

/*
 * class to represent tree structure - Splits on binary features
 */


public class Node {
	public int attribute;
	public int label;
	public Node[] children;

    /*
     * How many features do we want to choose from at each node in randomized
     * tree?
     */
    private int numFeatures(int total) {
        return (int)Math.sqrt(total) + 1;
        //return (int)Math.log(total) + 1;
    }
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
    int chooseAttribute(DataSet data,  HashSet<Integer> attributes,
            ArrayList<Integer> examples) {
        int bestAttr = -1;
        double bestGain = -1;
        int[] labelCount = new int[2];
        for (int ex : examples) {
            //Yay array index beauty...
            labelCount[data.trainLabel[ex]]++;
        }
        double setEntropy = entropy(labelCount[0], labelCount[1]);
        for (int attr : attributes) {
            //Set to something in the case that there is no gain
            //if (bestAttr == -1) { bestAttr = attr; }
            /*
             * [value][label] : so [0][0] + [0][1] would be the number of
             * examples with a value of 0 for the attributes.
             * [0][1] + [1][1] would be number of examples with label 1 in
             * examples.  Use this to calculate entropy and gain.
             */
            double[][] count = new double[data.attrVals[attr].length][2];
            for (int ex : examples) {
                //Yay array index beauty...
                count[data.trainEx[ex][attr]][data.trainLabel[ex]]++;
            }
            
            //Not needed, but using until I am sure my code works
            //(recalculates current entropy every time)
            //double gain = entropy(count[0][0] + count[1][0], count[0][1]
                    //+ count[1][1]);
            double gain = setEntropy;
            for (int val = 0; val < data.attrVals[attr].length; val++) {
                //Get number of examples with this attribute value
                gain -= ((count[val][0] + count[val][1]) / examples.size())
                            * entropy(count[val][0], count[val][1]);
            }

            if (gain >= bestGain) {
                bestAttr = attr;
                bestGain = gain;
            }
            //System.out.println(gain);
        }
        //System.out.println(bestAttr + "\t" + bestGain + "\t" + setEntropy
                //+ "\t" + attributes.size());
        return bestAttr;
    }

    /*
     * Accepts the dataset then an array of indices for remaining examples.
     * So each integer in examples is just the index of a training example
     * in data.trainEx.  Avoid making too many copies of the examples this
     * way.  attributes is a set of all remaining attributes to split on.
     * This is modified and passed to children by each node.
     */
    Node(DataSet data,  HashSet<Integer> attributes,
            ArrayList<Integer> examples, 
            boolean randomize, Random random) {

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

        /*
         * If randomization is on (i.e. being used in a random forest), then
         * we want to choose some random subset of features to choose best
         * split feature in.
         */
        if (randomize) {
            int numAttr = numFeatures(attributes.size());
            //More efficients ways to do this, but this works well enough
            HashSet<Integer> attrSample = new HashSet<Integer>(numAttr);
            for (int attr : attributes) {
                /*
                 * Add each with a probability of numAttr/number attr
                 * Also adds some variance so we slightly randomize the
                 * exact number
                 */
                if (random.nextInt(attributes.size()) < numAttr) {
                    attrSample.add(attr);
                }
            }
            //System.out.println(numAttr + " : " + attrSample.size());
            this.attribute = chooseAttribute(data, attrSample, examples);
        } else {
            this.attribute = chooseAttribute(data, attributes, examples);
        }

        /*No best attribute*/
        if (this.attribute == -1) {
            this.label = majority;
            return;
        }

        //Remove the attribute so it cannot be used again in child branches
        //Add it back in before returning
        attributes.remove(this.attribute);

        //Initialize list of examples that will be passed to children
        ArrayList<ArrayList<Integer>> childExamples = new
            ArrayList<ArrayList<Integer>>
                    (data.attrVals[this.attribute].length);
        for (int i = 0; i < data.attrVals[this.attribute].length; i++) {
            childExamples.add(new ArrayList<Integer>());
        }
        //for (ArrayList<Integer> l : childExamples) {
            //l = new ArrayList<Integer>();
        //}

        /*
         * Split examples based on the chosen attribute
         */
        for (int ex : examples) {
            childExamples.get(data.trainEx[ex][this.attribute]).add(ex);
        }

        children = new Node[data.attrVals[this.attribute].length];
        /*Create children trees*/
        for (int i = 0; i < data.attrVals[this.attribute].length; i++) {
            children[i] = new Node(data,
                                    attributes,
                                    childExamples.get(i),
                                    randomize,
                                    random);
            /*
             * Need to set child label if they don't have any examples to train
             * on
             */
            if (childExamples.get(i).size() == 0) {
                children[i].label = majority;
            }
        }
        attributes.add(this.attribute);
    }
}
