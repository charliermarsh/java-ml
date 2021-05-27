import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public abstract class DataSetFileReader {

	private String filename;
	private int line_count;
	private BufferedReader in;
	protected DataSet dataSet;
	
	public DataSetFileReader(String filename, DataSet dataSet) throws FileNotFoundException {
		super();
		this.filename = filename;
		open_file(filename);
		this.dataSet = dataSet;
	}
	
	public DataSetFileReader(DataSetFileReader fileReader) throws IOException {
		super();
		this.filename = fileReader.filename;
		this.line_count = fileReader.line_count;
		this.in = fileReader.in;
		this.dataSet = fileReader.dataSet;
		
		boolean isReadyToRead;
		try {
			isReadyToRead = fileReader.in.ready();
		} catch(IOException e) {
			String err = "Cannot read file \"" + filename + "\"";
			System.err.println(err);
			throw e;
		}
		if (!isReadyToRead) {
			String err = "Cannot read at " + location();
			System.err.println(err);
			throw new IOException(err);
		}
	}
	
	public void read() throws IOException {
		beforeRead();
		while (true) {
			String[] words = getWordsForALine();
			
			if ( isEndOfRead(words) )
				break;
			
			if ( isEmptyLine(words) )
				continue;
			
			if ( isLineError(words) ) {
				String err = getLineErrorMessage();
				System.err.println(err);
				throw new RuntimeException(err);
			}
			
			readEachLine(words);
		}
		afterRead();
	}

	public void close() throws IOException {
		line_count = -1;
		in.close();
	}

	protected abstract void beforeRead();

	protected abstract boolean isLineError(String[] words);

	protected abstract String getLineErrorMessage();

	protected abstract void readEachLine(String[] words);

	protected abstract void afterRead();
	
	protected boolean isEndOfRead(String[] words) {
		boolean isEndOfFile = (words == null);
		return isEndOfFile;
	}

	protected String location() {
		return "line " + line_count + " in file " + filename;
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

	private boolean isEmptyLine(String[] words) {
		return words.length == 1 && words[0].equals("");
	}
}
