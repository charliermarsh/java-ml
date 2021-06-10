import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class MultiLayerNeuralNetTest {
	public MultiLayerNeuralNet c;
	@BeforeEach
	void createMultiLayerNueralNet() throws Exception{
    	String filestem = "data/lnn_test_census";
    	DataSetInput input = new FileInput(filestem);
    	DataSet d = new BinaryDataSet(input);
    	Activation a = new Sigmoid();
    	
    	c = new MultiLayerNeuralNet(d, a);
		
	}
	/**
	* Purpose: Valid range of getIndex function
	* Input: getIndex (0,0),(0,N_i-1),(1,0),(1,N_h-1),(2,0),(2,N_o-1)
	* Expected:
	* return Succcess
	*/
	@Test
	void test_getIdx_Valid(){
		int inputLayerSize = c.getLayerSize(0);
		int hiddenLayerSize = c.getLayerSize(1);
		int outputLayerSize = c.getLayerSize(2);
		
		assertEquals(c.getIdx(0,0), 0);
		assertEquals(c.getIdx(0,inputLayerSize-1), inputLayerSize-1);

		assertEquals(c.getIdx(1,0), inputLayerSize);
		assertEquals(c.getIdx(1,hiddenLayerSize-1), inputLayerSize + hiddenLayerSize - 1);
		
		assertEquals(c.getIdx(2,0), inputLayerSize + hiddenLayerSize);
		assertEquals(c.getIdx(2,outputLayerSize-1), inputLayerSize + hiddenLayerSize + outputLayerSize - 1);
	}
	/**
	* Purpose: Invalid range of getIndex function
	* Input: getIndex (0,-1),(0,N_i),(1,-1),(1,N_h),(2,-1),(2,N_o),(-1,0),(3,0)
	* Expected:
	* return -1
	*/
	@Test
	void test_getIdx_Invalid(){
		int inputLayerSize = c.getLayerSize(0);
		int hiddenLayerSize = c.getLayerSize(1);
		int outputLayerSize = c.getLayerSize(2);
		
		assertEquals(c.getIdx(0,-1), -1);
		assertEquals(c.getIdx(0,inputLayerSize), -1);

		assertEquals(c.getIdx(1,-1), -1);
		assertEquals(c.getIdx(1,hiddenLayerSize), -1);
		
		assertEquals(c.getIdx(2,-1), -1);
		assertEquals(c.getIdx(2,outputLayerSize), -1);

		assertEquals(c.getIdx(-1,0), -1);
		assertEquals(c.getIdx(3,0), -1);
	}
}
