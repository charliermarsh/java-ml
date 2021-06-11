import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class SigmoidTest {
	/**
	* Purpose: Calculate sigmoid function
	* Input: getActivation (2.0) 
	* Expected:
	* return Succcess
	*/
	@Test
	void test_getActivation() {
		Sigmoid sigmoid = new Sigmoid();
		double test = sigmoid.getActivation(2.0);
		boolean result = (0.88079707797780 < test) && (test < 0.88079707797790) ;
		System.out.println(test);
		assertTrue(result);
	}
	/**
	* Purpose: Calculate sigmoid derivation function
	* Input: getDerivation (2.0) 
	* Expected:
	* return Succcess
	*/
	@Test
	void test_getDerivation() {
		Sigmoid sigmoid = new Sigmoid();
		double test = sigmoid.getDerivation(2.0);
		boolean result = (0.10499358540350 < test) && (test < 0.10499358540351) ;
		System.out.println(test);
		assertTrue(result);
	}
}
