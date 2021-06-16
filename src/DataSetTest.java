import static org.junit.Assert.*;

import java.util.Arrays;

import org.junit.BeforeClass;
import org.junit.Test;

public class DataSetTest {
	private static DataSet dataSet;

	@BeforeClass
	public static void setUp() throws Exception {
		String filestem = "./data/census"; 
		
		DataSetInput input = new FileInput(filestem);
		dataSet = new DataSet(input);
	}
	
	@Test
	public void testDataSet() {
		DataSet dataSet = new DataSet();
		
		assertEquals(0, dataSet.numAttrs);
		assertEquals(dataSet.numAttrs, dataSet.attrName.length);
		assertEquals(dataSet.numAttrs, dataSet.attrVals.length);
		
		assertEquals(2, dataSet.className.length);
		assertEquals("0", dataSet.className[0]);
		assertEquals("1", dataSet.className[1]);
		
		assertEquals(0, dataSet.numTrainExs);
		assertEquals(dataSet.numTrainExs, dataSet.trainEx.length);
		assertEquals(dataSet.numTrainExs, dataSet.trainLabel.length);
		
		assertEquals(0, dataSet.numTestExs);
		assertEquals(dataSet.numTestExs, dataSet.testEx.length);
	}

	@Test
	public void testDataSetDataSetInputAndIsNumericAttribute() throws Exception {		
		// check attribute name and values
		String attrName[] = {"age", "workclass", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"};
		String attrVals[][] = {
				{"numeric"},
				{"Private",  "Self-emp-not-inc",  "Self-emp-inc",  "Federal-gov",  "Local-gov",  "State-gov",  "Without-pay",  "Never-worked"},
				{"Bachelors",  "Some-college",  "11th",  "HS-grad",  "Prof-school",  "Assoc-acdm",  "Assoc-voc",  "9th",  "7th-8th",  "12th",  "Masters",  "1st-4th",  "10th",  "Doctorate",  "5th-6th",  "Preschool"},
				{"numeric"},
				{"Married-civ-spouse",  "Divorced",  "Never-married",  "Separated",  "Widowed",  "Married-spouse-absent",  "Married-AF-spouse"},
				{"Tech-support",  "Craft-repair",  "Other-service",  "Sales",  "Exec-managerial",  "Prof-specialty",  "Handlers-cleaners",  "Machine-op-inspct",  "Adm-clerical",  "Farming-fishing",  "Transport-moving",  "Priv-house-serv",  "Protective-serv",  "Armed-Forces"},
				{"Wife",  "Own-child",  "Husband",  "Not-in-family",  "Other-relative",  "Unmarried"},
				{"White",  "Asian-Pac-Islander",  "Amer-Indian-Eskimo",  "Other",  "Black"},
				{"Female",  "Male"},
				{"numeric"},
				{"numeric"},
				{"numeric"},
				{"United-States",  "Cambodia",  "England",  "Puerto-Rico",  "Canada",  "Germany",  "Outlying-US(Guam-USVI-etc)",  "India",  "Japan",  "Greece",  "South",  "China",  "Cuba",  "Iran",  "Honduras",  "Philippines",  "Italy",  "Poland",  "Jamaica",  "Vietnam",  "Mexico",  "Portugal",  "Ireland",  "France",  "Dominican-Republic",  "Laos",  "Ecuador",  "Taiwan",  "Haiti",  "Columbia",  "Hungary",  "Guatemala",  "Nicaragua",  "Scotland",  "Thailand",  "Yugoslavia",  "El-Salvador",  "Trinadad&Tobago",  "Peru",  "Hong",  "Holand-Netherlands"}
		};
		assertEquals(attrName.length, dataSet.numAttrs);
		assertTrue(Arrays.equals(attrName, dataSet.attrName));
		assertEquals(dataSet.numAttrs, dataSet.attrVals.length);
		for(int i=0; i<attrVals.length; ++i) {
			if( attrVals[i][0] == "numeric" ) {
				assertTrue(dataSet.isNumericAttribute(i));
			}
			else {
				assertFalse(dataSet.isNumericAttribute(i));
				assertTrue(Arrays.equals(attrVals[i], dataSet.attrVals[i]));
			}
		}
		
		// check class name
		String className[] = {">50K", "<=50K"};
		assertTrue(Arrays.equals(className, dataSet.className));

		// check train examples
		int trainExampleIndex = 0;
		int aTrainExample[] = {60, 1, 3, 9, 0, 4, 2, 0, 1, 0, 0, 50, 0};
		int aTrainLabel = 0; 
		assertEquals(2000, dataSet.numTrainExs);
		assertEquals(dataSet.numTrainExs, dataSet.trainEx.length);
		assertEquals(dataSet.numTrainExs, dataSet.trainLabel.length);
		assertTrue(Arrays.equals(aTrainExample, dataSet.trainEx[trainExampleIndex]));
		assertEquals(aTrainLabel, dataSet.trainLabel[trainExampleIndex]);
		
		// check test examples
		int testExampleIndex = 0;
		int aTestExample[] = {33, 0, 0, 13, 2, 8, 3, 0, 0, 0, 0, 40, 0};
		assertEquals(4000, dataSet.numTestExs);
		assertTrue(Arrays.equals(aTestExample, dataSet.testEx[testExampleIndex]));
	}
	
	@Test
	public void testGetContinuousValues() throws Exception {
		int numericValues[][] = dataSet.getContinuousValues();
		
		for(int i=0; i<dataSet.numAttrs; ++i) {
			if( !dataSet.isNumericAttribute(i) ) { 
				assertTrue(numericValues[i] == null);
				continue;
			}
			
			// check ascending order without duplication
			for(int j=1; j<numericValues[i].length; ++j) {
				assertTrue(numericValues[i][j-1] < numericValues[i][j]);
			}
			
			// Make sure it contains all values
			for(int j=0; j<dataSet.numTrainExs; ++j) {
				int aNumericValue = dataSet.trainEx[j][i];
				assertTrue(Arrays.binarySearch(numericValues[i], aNumericValue) >= 0);
			}
			for(int j=0; j<dataSet.numTestExs; ++j) {
				int aNumericValue = dataSet.testEx[j][i];
				assertTrue(Arrays.binarySearch(numericValues[i], aNumericValue) >= 0);
			}
		}
	}

}
