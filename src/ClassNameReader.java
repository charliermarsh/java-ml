import java.io.FileNotFoundException;
import java.io.IOException;

public class ClassNameReader extends DataSetFileReader {

	public ClassNameReader(String filename, DataSet dataSet) throws FileNotFoundException {
		super(filename, dataSet);
	}

	public ClassNameReader(DataSetFileReader fileReader) throws IOException {
		super(fileReader);
	}

	@Override
	protected void beforeRead() {
	}

	@Override
	protected boolean isLineError(String[] words) {
		boolean isEnoughNumber = (words.length == 2);
		return !isEnoughNumber;
	}
	
	@Override
	protected String getLineErrorMessage() {
		String errorMessage = "expected two class names at " + location();
		return errorMessage;
	}
	
	@Override
	protected void readEachLine(String[] words) {
		dataSet.className = words;
	}
	
	@Override
	protected void afterRead() {
	}

	@Override
	protected boolean isEndOfRead(String[] words) {
		boolean result = super.isEndOfRead(words);
		boolean isClassNameFilled = (dataSet.className != null);
		result = result || isClassNameFilled;
		return result;
	}

}
