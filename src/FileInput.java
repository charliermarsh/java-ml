import java.io.FileNotFoundException;

public class FileInput implements DataSetInput {

	private static final String ATTRIBUTE_INFO_FILE_EXTENSION = ".names";
	private static final String TRAIN_EXAMPLE_FILE_EXTENSION = ".train";
	private static final String TEST_EXAMPLE_FILE_EXTENSION = ".test";
	private String filestem;
	
	public FileInput(String filestem) {
		super();
		this.filestem = filestem;
	}
	
	public void readTo(DataSet dataSet) throws Exception {

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
