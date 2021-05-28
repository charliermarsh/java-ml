import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

public class LabeledExampleReader extends ExampleReader {
	ArrayList<Integer> labels = new ArrayList<Integer>();

	public LabeledExampleReader(String filename, DataSet dataSet) throws FileNotFoundException {
		super(filename, dataSet);
	}

	public LabeledExampleReader(DataSetFileReader fileReader) throws IOException {
		super(fileReader);
	}

	@Override
	protected boolean isLineError(String[] words) {
		return words.length != dataSet.numAttrs + 1;
	}
	
	@Override
	protected void readEachLine(String[] words) {
		super.readEachLine(words);
		
		int aLabel = convertToALabel(dataSet, words);
		labels.add( (Integer)aLabel );
	}

	@Override
	protected void afterRead() {
		dataSet.numTrainExs = examples.size();
		dataSet.trainEx = new int[0][];
		dataSet.trainEx = (int[][]) examples.toArray(dataSet.trainEx);
		dataSet.trainLabel = toArray(labels);
	}

	private int convertToALabel(DataSet dataSet, String[] words) {
		int aLabel;
		String className = words[dataSet.numAttrs];
		if (className.equals(dataSet.className[0])) {
			aLabel = 0;
		} else if (className.equals(dataSet.className[1])) {
			aLabel = 1;
		} else {
			String err = "unrecognized label at " + location();
			System.err.println(err);
			throw new RuntimeException(err);
		}
		return aLabel;
	}
	
	private int[] toArray(ArrayList<Integer> arrayList) {
		int[] array = new int[arrayList.size()];
		for (int i = 0; i < arrayList.size(); i++) {
			array[i] = arrayList.get(i).intValue();
		}
		return array;
	}
}
