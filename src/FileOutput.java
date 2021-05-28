import java.io.BufferedOutputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;

public class FileOutput {

	private static final String FILE_EXTENSION = ".testout";
	private PrintStream out;
	
	public FileOutput() {
		super();
		this.out = null;
	}
	
	public FileOutput(PrintStream out) {
		super();
		this.out = out;
	}
	
	public void write(DataSet dataSet, Classifier classifier) {
		assert out != null;
		header(classifier);
		body(dataSet, classifier);
	}
	
	public void write(DataSet dataSet, Classifier classifier, String filestem) throws FileNotFoundException {
		openFile(filestem);
		
		write(dataSet, classifier);

		out.close();
	}

	private void openFile(String filestem) throws FileNotFoundException {
		String fileName = filestem + FILE_EXTENSION;
		try {
			out = new PrintStream(new BufferedOutputStream(new FileOutputStream(fileName)));
		} catch (FileNotFoundException e) {
			System.err.println("Cannot open file " + fileName);
			throw e;
		}
	}

	private void header(Classifier classifier) {
		out.println(classifier.author());
		out.println(".");
		out.println(classifier.algorithmDescription());
		out.println(".");
	}

	private void body(DataSet dataSet, Classifier classifier) {
		for (int i = 0; i < dataSet.numTestExs; i++) {
			int[] aExample = dataSet.testEx[i];
			int classIndex = classifier.predict(aExample);
			String className = dataSet.className[classIndex];
			out.println(className);
		}
	}
}
