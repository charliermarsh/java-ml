import static org.junit.Assert.assertEquals;

import org.junit.Test;

import java.lang.reflect.Method;

public class kNNTest {
	/*
	 * Purpose: Test knn class member variable initialization in knn constructor
	 * Input: kNN Create kNN object
	 * Expected:
	 * 		getDataSet() = dataset
	 *		getStrategy() = strategy
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
	 * Input: getDistance get distance between {1,0,0,1} and {1,0,1,0}
	 * Expected:
	 * 		return 2
	 */
	@Test
	public void testgetDistance() throws Exception {
		DataSetInput input = new FileInput("./data/knn_test_dna");
		DataSet dataset = new BinaryDataSet(input);
		Strategy strategy = new Strategy(new EuclideanDistance(), new kFoldCrossValidation());
		kNN knn = new kNN(dataset, strategy);
		
		Method method = knn.getClass().getDeclaredMethod("getDistance", int[].class, int[].class); 
		method.setAccessible(true);
		
		int[] vector1 = {1,0,0,1};
		int[] vector2 = {1,0,1,0};
		
		double distance = (double)method.invoke(knn, vector1, vector2);
		assertEquals(distance, 2, 0.00001);
	}
	
	
	
}
