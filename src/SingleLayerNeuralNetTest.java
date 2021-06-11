import static org.junit.jupiter.api.Assertions.*;

import org.junit.Before;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class SingleLayerNeuralNetTest {
	public DataSet d;
	public Classifier c;
	@BeforeEach
	void createSingleLayerNeuralNet() throws Exception{
    	String filestem = "data/lnn_test_census";
    	DataSetInput input = new FileInput(filestem);
    	d = new BinaryDataSet(input);
    	c = new SingleLayerNeuralNet(d);
	}
	/**
	* Purpose: Check if prediction is equal to label
	* Input: DataSet d
	* Expected:
	* return Success
	*/
	@Test
	void test_SingleLayerNeuralNet() {
		String[] classLabel = {">50K","<=50K","<=50K",">50K","<=50K"};
		for (int i = 0; i < this.d.numTestExs; i++) {
			int[] aExample = this.d.testEx[i];
			int classIndex = this.c.predict(aExample);
			String className = this.d.className[classIndex];
			assertEquals(classLabel[i], className);
		}
	}

}
