import static org.junit.Assert.*;

import java.util.HashSet;
import java.util.Random;

import org.junit.After;
import org.junit.Test;

public class TreeTest {
	/**
	* Purpose:test whether the learning algorithm is well applied using a tree with a prediction ratio of 100%.
	* Input: DecisionTree Create DecisionTree
	* Expected:
	* 			100 * correct / crossEx.length (prediction ratio) = 100
	*/
	@Test
	public void testDecisionTree() throws Exception {
        String filestem = "data/testTreedata";
        
        DiscreteDataSet d = new DiscreteDataSet(filestem);

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

        d.numTrainExs = dEx.length;
        d.trainEx = dEx;
        d.trainLabel = dLabel;


        Classifier c = new DecisionTree(d, false);

        int correct = 0;
        for (int ex = 0; ex < crossEx.length; ex++) {
            if (c.predict(crossEx[ex]) == crossLabel[ex])
                correct++;
        }
	
        assertEquals((100*correct / crossEx.length),100);

	}
		/**
	* Purpose:test whether the learning algorithm is well applied using a tree with a prediction ratio of 100% in other constructor .
	* Input: DecisionTree(constructor) Create DecisionTree
	* Expected:
	* 			100 * correct / crossEx.length (prediction ratio) = 100
	*/
	@Test
	public void testDecisionTreeConstructor1() throws Exception {
        String filestem = "data/testTreedata";
        
        DiscreteDataSet d = new DiscreteDataSet(filestem);

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

        d.numTrainExs = dEx.length;
        d.trainEx = dEx;
        d.trainLabel = dLabel;
        
        HashSet<Integer> attributes = new HashSet<Integer>(d.numAttrs);
    	for (int i = 0; i < d.numAttrs; i++) { attributes.add(i); }
        Classifier c = new DecisionTree(d, attributes, false);

        int correct = 0;
        for (int ex = 0; ex < crossEx.length; ex++) {
            if (c.predict(crossEx[ex]) == crossLabel[ex])
                correct++;
        }
	
        assertEquals((100*correct / crossEx.length),100);

	}

	/**
	* Purpose:test whether the learning algorithm is well applied using a tree with a prediction ratio of 100% in other constructor .
	* Input: DecisionTree(constructor) Create DecisionTree
	* Expected:
	* 			100 * correct / crossEx.length (prediction ratio) = 100
	*/
	@Test
	public void testDecisionTreeConstructor2() throws Exception {
        String filestem = "data/testTreedata";
        
        DiscreteDataSet d = new DiscreteDataSet(filestem);

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

        d.numTrainExs = dEx.length;
        d.trainEx = dEx;
        d.trainLabel = dLabel;
        
        HashSet<Integer> attributes = new HashSet<Integer>(d.numAttrs);
    	for (int i = 0; i < d.numAttrs; i++) { attributes.add(i); }
        ArrayList<Integer> examples = new ArrayList<Integer>(d.numTrainExs);
        for (int i = 0; i < d.numTrainExs; i++) { examples.add(i); }
        Classifier c = new DecisionTree(d, attributes, examples, false);

        int correct = 0;
        for (int ex = 0; ex < crossEx.length; ex++) {
            if (c.predict(crossEx[ex]) == crossLabel[ex])
                correct++;
        }
	
        assertEquals((100*correct / crossEx.length),100);

	}
	
	/**
	* Purpose:test Adaboost using a tree with a prediction ratio of 100%.
	* Input: Adaboost Create Adaboost
	* Expected:
	* 			100 * correct / crossEx.length (prediction ratio) = 100
	*/
	@Test
	public void testAdaboost() throws Exception {
        String filestem = "data/testTreedata";

        DiscreteDataSet d = new DiscreteDataSet(filestem);

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

        int crossSize = d.numTrainExs/4;

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

        d.numTrainExs = dEx.length;
        d.trainEx = dEx;
        d.trainLabel = dLabel;


        Classifier c = new Adaboost(d,3) ;

        int correct = 0;
        for (int ex = 0; ex < crossEx.length; ex++) {
            if (c.predict(crossEx[ex]) == crossLabel[ex])
                correct++;
        }

        assertEquals((100*correct / crossEx.length),100);

    }
	
		/**
	* Purpose:test DecisionForest using a tree with a prediction ratio of 100%.
	* Input: DecisionForest Create DecisionForest
	* Expected:
	* 			No, No, No, No, No, No,
				No, Yes, Yes, Yes, Yes,
				Yes, Yes, Yes, Yes, No,
				No, No, No, No, No, Yes,
				Yes, Yes, Yes, Yes, Yes
				in testTreedata.testout
	*/
	@Test
	public void testDecisionForest() throws Exception {


        String filestem = "data/testTreedata";

        /*
         * Create a cross validation set - just takes the last crossSize
         * elements of the set as a cross set.
         */
        DiscreteDataSet d = new DiscreteDataSet(filestem);

        System.out.println("Training classifier on " + d.numTrainExs
                + " examples");

        Classifier c = new DecisionForest(d,5);

        System.out.println("Running on test set...");
        d.printTestPredictions(c, filestem);
	}
}
