import static org.junit.Assert.assertEquals;

import org.junit.Test;

import java.lang.reflect.Method;

public class kNNTest {
	/*
	 * Purpose: Test knn class member variable initialization in knn constructor
	 * Input: kNN Create kNN object 
	 * Expected: 
	 * 		dataset = getDataSet()
	 * 		strategy = getStrategy()
	 */
	@Test
	public void testkNN() throws Exception {
		DataSetInput input = new FileInput("./data/knn_test_dna");
		DataSet dataset = new BinaryDataSet(input);
		Strategy strategy = new Strategy(new EuclideanDistance(), new kFoldCrossValidation());
		kNN knn = new kNN(dataset, strategy);

		assertEquals(dataset, knn.getDataSet());
		assertEquals(strategy, knn.getStrategy());
	}

	/*
	 * Purpose: Calculates the distance between two vectors 
	 * Input: getDistance Get distance between {1,0,0,1} and {1,0,1,0} 
	 * Expected: 
	 * 		return 2
	 */
	@Test
	public void testGetDistance() throws Exception {
		DataSetInput input = new FileInput("./data/knn_test_dna");
		DataSet dataset = new BinaryDataSet(input);
		Strategy strategy = new Strategy(new EuclideanDistance(), new kFoldCrossValidation());
		kNN knn = new kNN(dataset, strategy);

		Method method = knn.getClass().getDeclaredMethod("getDistance", int[].class, int[].class);
		method.setAccessible(true);

		int[] vector1 = {1,0,0,1};
		int[] vector2 = {1,0,1,0};
		
		boolean[] isEliminatedAttr = new boolean[4];
		knn.setIsEliminatedAttr(isEliminatedAttr);
		
		double distance = (double) method.invoke(knn, vector1, vector2);
		assertEquals(distance, 2, 0.00001);
	}

	/*
	 * Purpose: Test for calcError(), a method that calculates the error of the predicted value.
	 * Input: calcError Calculates the error over a labeled data set(knn_test_dna.train) using 1-nearest indices(kNNindices) 
	 * Expected: 
	 * 		return 0
	 */
	@Test
	public void testCalcError() throws Exception {
		DataSetInput input = new FileInput("./data/knn_test_dna");
		DataSet dataset = new BinaryDataSet(input);
		Strategy strategy = new Strategy(new EuclideanDistance(), new kFoldCrossValidation());
		kNN knn = new kNN(dataset, strategy);
		
		Method method = knn.getClass().getDeclaredMethod("calcError", int[][].class);
		method.setAccessible(true);
		
		double[] instanceWeights = new double[10];
		for (int i = 0; i < instanceWeights.length; i++)
			instanceWeights[i] = 1.0;
		
		knn.setInstanceWeights(instanceWeights);
		
		int[][] kNNindices = { {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9} };
		double error = (double) method.invoke(knn, (Object)kNNindices);
		assertEquals(error, 0, 0.00001);
	}
	
	/*
	 * Purpose: Test for kNearest(), a method that calculates the indices of the k nearest training examples in data set 
	 * Input: kNearest Calculate the two indices closest to the dna sequence 'GGGGGG'
	 * Expected:
	 * 		return {0, 1}
	 */
	@Test
	public void testkNearest() throws Exception {
		DataSetInput input = new FileInput("./data/knn_test_dna");
		DataSet dataset = new BinaryDataSet(input);
		Strategy strategy = new Strategy(new EuclideanDistance(), new kFoldCrossValidation());
		kNN knn = new kNN(dataset, strategy);
		
		Method method = knn.getClass().getDeclaredMethod("kNearest", int.class, int[].class);
		method.setAccessible(true);
		
		int[] ex = {0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0}; // G G G G G G
		int k = 2;
		
		int[] kNNindices = (int[])method.invoke(knn, k,  ex);
		int[] answer = {0, 1};
		for(int i = 0; i < k; i++) {
			assertEquals(kNNindices[i], answer[i]);
		}
	}
	
	

}
