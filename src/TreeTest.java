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
}
