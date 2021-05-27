import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

public class ExampleReader extends DataSetFileReader {
	private static final int NOT_FOUND = -1;
	ArrayList<int[]> examples = new ArrayList<int[]>();
	
	public ExampleReader(String filename, DataSet dataSet) throws FileNotFoundException {
		super(filename, dataSet);
	}

	public ExampleReader(DataSetFileReader fileReader) throws IOException {
		super(fileReader);
	}

	@Override
	protected void beforeRead() {

	}

	@Override
	protected boolean isLineError(String[] words) {
		return words.length != dataSet.numAttrs;
	}

	@Override
	protected String getLineErrorMessage() {
		String errorMessage = "wrong number of tokens at " + location();
		return errorMessage;
	}

	@Override
	protected void readEachLine(String[] words) {
		int[] aExample = convertToAExample(dataSet, words);
		examples.add(aExample);
	}

	@Override
	protected void afterRead() {
		dataSet.numTestExs = examples.size();
		dataSet.testEx = new int[0][];
		dataSet.testEx = (int[][]) examples.toArray(dataSet.testEx);
	}

	private int[] convertToAExample(DataSet dataSet, String[] words) {
		int aExample[] = new int[dataSet.numAttrs];
		for (int attrNum = 0; attrNum < dataSet.numAttrs; attrNum++) {
			if (dataSet.isNumericAttribute(attrNum)) {
				try {
					aExample[attrNum] = Integer.parseInt(words[attrNum]);
				} catch (NumberFormatException e) {
					String err = "Expected integer in field " + (attrNum + 1) + " at " + location();
					System.err.println(err);
					throw e;
				}
			} else {
				int valueNum = findAttributeValueNumber(dataSet.attrVals[attrNum], words[attrNum]);
				if( valueNum == NOT_FOUND ) {
					String err = "bad attribute value in field " + (attrNum + 1) + " at " + location();
					System.err.println(err);
					throw new RuntimeException(err);
				}
				aExample[attrNum] = valueNum;
			}
		}
		return aExample;
	}
	
	private int findAttributeValueNumber(String[] values, String value) {
		int valueNum;
		for (valueNum = 0; valueNum < values.length; valueNum++) {
			if( values[valueNum].equals(value) )
				break;
		}
		boolean isFound = (valueNum < values.length);
		if ( !isFound ) {
			return NOT_FOUND;
		}
		return valueNum;
	}
}
