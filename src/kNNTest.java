import static org.junit.Assert.assertEquals;

import org.junit.Test;

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
	
	
}
