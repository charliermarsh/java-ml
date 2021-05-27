import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class AttributeNameReader extends DataSetFileReader {
	ArrayList<String[]> attributes = new ArrayList<String[]>();

	public AttributeNameReader(String filename, DataSet dataSet) throws FileNotFoundException {
		super(filename, dataSet);
	}

	public AttributeNameReader(DataSetFileReader fileReader) throws IOException {
		super(fileReader);
	}

	@Override
	protected void beforeRead() {
	}

	@Override
	protected boolean isLineError(String[] words) {
		return words.length <= 1;
	}

	@Override
	protected String getLineErrorMessage() {
		String errorMessage = "expected attribute description at " + location();
		return errorMessage;
	}

	@Override
	protected void readEachLine(String[] words) {
		attributes.add(words);
	}

	@Override
	protected void afterRead() {
		dataSet.numAttrs = attributes.size();
		dataSet.attrName = new String[dataSet.numAttrs];
		dataSet.attrVals = new String[dataSet.numAttrs][];

		for (int i = 0; i < dataSet.numAttrs; i++) {
			String[] words = attributes.get(i);
			
			dataSet.attrName[i] = words[0];
			
			if ( isNumericAttribute(words[1]) ) 
				dataSet.setAttributeNumeric(i);
			else 
				dataSet.attrVals[i] = Arrays.copyOfRange(words, 1, words.length);
		}
	}
	
	private boolean isNumericAttribute(String attributeValue) {
		return attributeValue.equals("numeric");
	}

}
