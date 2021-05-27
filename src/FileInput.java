import java.io.FileNotFoundException;
import java.io.IOException;

public class FileInput {
	private static final String ATTRIBUTE_INFO_FILE_EXTENSION = ".names";
	private static final String TRAIN_EXAMPLE_FILE_EXTENSION = ".train";
	private static final String TEST_EXAMPLE_FILE_EXTENSION = ".test";
	
	public void read(DataSet dataSet, String filestem) throws FileNotFoundException, IOException {
		DataSetFileReader classNameReader 
			= new ClassNameReader(filestem+ATTRIBUTE_INFO_FILE_EXTENSION, dataSet);
		classNameReader.read();
		
		DataSetFileReader attributeReader 
			= new AttributeNameReader(classNameReader);
		attributeReader.read();
		attributeReader.close();
	
		DataSetFileReader trainExampleReader 
			= new LabeledExampleReader(filestem+TRAIN_EXAMPLE_FILE_EXTENSION, dataSet);
		trainExampleReader.read();
		trainExampleReader.close();
		
		DataSetFileReader testExampleReader;
		try {
			testExampleReader 
				= new ExampleReader(filestem+TEST_EXAMPLE_FILE_EXTENSION, dataSet);
		} catch (FileNotFoundException e) {
			System.err.print("Continuing without test file...\n");
			dataSet.numTestExs = 0;
			dataSet.testEx = new int[0][];
			return;
		}
		testExampleReader.read();
		testExampleReader.close();
	}
}
