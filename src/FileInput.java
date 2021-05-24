
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class FileInput {
	private static final int NOT_FOUND = -1;
	private static final String ATTRIBUTE_INFO_FILE_EXTENSION = ".names";
	private static final String TRAIN_EXAMPLE_FILE_EXTENSION = ".train";
	private static final String TEST_EXAMPLE_FILE_EXTENSION = ".test";
	private String filename;
	private int line_count;
	private BufferedReader in;
	
	public void read(DataSet dataSet, String filestem) throws FileNotFoundException, IOException {
		open_file(filestem + ATTRIBUTE_INFO_FILE_EXTENSION);
		readClassInfomation(dataSet);
		readAttributeInformation(dataSet);
		in.close();
		
		open_file(filestem + TRAIN_EXAMPLE_FILE_EXTENSION);
		readTrainExamples(dataSet);
		in.close();
		
		try {
			open_file(filestem + TEST_EXAMPLE_FILE_EXTENSION);
		} catch (FileNotFoundException e) {
			System.err.print("Continuing without test file...\n");
			dataSet.numTestExs = 0;
			dataSet.testEx = new int[0][];
			return;
		}
		readTestExamples(dataSet);
		in.close();
	}

	private void readClassInfomation(DataSet dataSet) throws IOException {
		while (true) {
			String[] words = getWordsForALine();
			
			if ( dataSet.className != null || isEndOfFile(words) )
				break;
			
			if ( isEmptyLine(words) )
				continue;
			
			if (words.length != 2) {
				String err = "expected two class names at " + location();
				System.err.println(err);
				throw new RuntimeException(err);
			}
			
			dataSet.className = words;
		}
	}

	private void readAttributeInformation(DataSet dataSet) throws IOException {
		ArrayList<String[]> attributes = new ArrayList<String[]>();
		while (true) {
			String[] words = getWordsForALine();
			
			if ( isEndOfFile(words) )
				break;
			
			if ( isEmptyLine(words) )
				continue;
			
			if (words.length <= 1) {
				String err = "expected attribute description at " + location();
				System.err.println(err);
				throw new RuntimeException(err);
			}
			
			attributes.add(words);
		}

		dataSet.numAttrs = attributes.size();
		dataSet.attrName = new String[dataSet.numAttrs];
		dataSet.attrVals = new String[dataSet.numAttrs][];

		for (int i = 0; i < dataSet.numAttrs; i++) {
			String[] words = attributes.get(i);
			
			dataSet.attrName[i] = words[0];
			
			if ( isNumericAttribute(words[1]) ) 
				dataSet.attrVals[i] = null;
			else 
				dataSet.attrVals[i] = Arrays.copyOfRange(words, 1, words.length);
		}
	}

	private void readTrainExamples(DataSet dataSet) throws IOException {
		ArrayList<int[]> examples = new ArrayList<int[]>();
		ArrayList<Integer> labels = new ArrayList<Integer>();
		
		while (true) {
			String[] words = getWordsForALine();
			
			if ( isEndOfFile(words) )
				break;
			
			if ( isEmptyLine(words) )
				continue;
			
			if (words.length != dataSet.numAttrs + 1) {
				String err = "wrong number of tokens at " + location();
				System.err.println(err);
				throw new RuntimeException(err);
			}

			int[] aExample = convertToAExample(dataSet, words);
			examples.add(aExample);
			
			int aLabel = convertToALabel(dataSet, words);
			labels.add( (Integer)aLabel );
		}
		
		dataSet.numTrainExs = examples.size();
		dataSet.trainEx = new int[0][];
		dataSet.trainEx = (int[][]) examples.toArray(dataSet.trainEx);
		dataSet.trainLabel = toArray(labels);
	}

	private void readTestExamples(DataSet dataSet) throws IOException {
		ArrayList<int[]> examples = new ArrayList<int[]>();

		while (true) {
			String[] words = getWordsForALine();
			
			if ( isEndOfFile(words) )
				break;
			
			if ( isEmptyLine(words) )
				continue;
			
			if (words.length != dataSet.numAttrs) {
				String err = "wrong number of tokens at " + location();
				System.err.println(err);
				throw new RuntimeException(err);
			}

			int[] aExample = convertToAExample(dataSet, words);
			examples.add(aExample);
		}

		dataSet.numTestExs = examples.size();
		dataSet.testEx = new int[0][];
		dataSet.testEx = (int[][]) examples.toArray(dataSet.testEx);
	}
	
	private int[] toArray(ArrayList<Integer> arrayList) {
		int[] array = new int[arrayList.size()];
		for (int i = 0; i < arrayList.size(); i++) {
			array[i] = arrayList.get(i).intValue();
		}
		return array;
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

	private int[] convertToAExample(DataSet dataSet, String[] words) {
		int aExample[] = new int[dataSet.numAttrs];
		for (int attrNum = 0; attrNum < dataSet.numAttrs; attrNum++) {
			if (isNumericAttribute(dataSet, attrNum)) {
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

	public static boolean isNumericAttribute(DataSet dataSet, int attribute) {
		return dataSet.attrVals[attribute] == null;
	}
	
	private boolean isNumericAttribute(String attributeValue) {
		return attributeValue.equals("numeric");
	}

	private boolean isEmptyLine(String[] words) {
		return words.length == 1 && words[0].equals("");
	}

	private boolean isEndOfFile(String[] words) {
		return words == null;
	}

	private String[] getWordsForALine() throws IOException {
		String line;
		line = read_line();
		if (line == null)
			return null;
		line = line.trim();
		String[] words = line.split("\\s+");
		return words;
	}

	private void open_file(String filename) throws FileNotFoundException {
		BufferedReader in;

		this.filename = filename;
		this.line_count = 0;

		try {
			in = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException e) {
			System.err.print("File " + filename + " not found.\n");
			throw e;
		}
		this.in = in;
	}

	private String read_line() throws IOException {
		String line;

		line_count++;

		try {
			line = in.readLine();
		} catch (IOException e) {
			System.err.println("Error reading " + location());
			throw e;
		}
		return line;
	}

	private String location() {
		return "line " + line_count + " in file " + filename;
	}
}
