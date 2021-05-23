import java.io.*;
import java.util.ArrayList;

public class FileInput {
	private String filename;
	private int line_count;
	private BufferedReader in;

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
			System.err.println("Error reading line " + line_count + " in file " + filename);
			throw e;
		}
		return line;
	}


	public void readAttributeInformation(DataSet dataSet, String filestem) throws FileNotFoundException, IOException {
		open_file(filestem + ".names");
		ArrayList<String[]> attr_list = new ArrayList<String[]>();

		String line;
		while ((line = read_line()) != null) {
			line = line.trim();
			String[] words = line.split("\\s+");
			if (line.equals(""))
				continue;

			if (dataSet.className == null) {
				if (words.length != 2) {
					String err = "expected two class names at line " + line_count + " in file " + filename;
					System.err.println(err);
					throw new RuntimeException(err);
				}
				dataSet.className = words;
			} else {
				if (words.length <= 1) {
					String err = "expected attribute description at line " + line_count + " in file " + filename;
					System.err.println(err);
					throw new RuntimeException(err);
				}
				attr_list.add(words);
				dataSet.numAttrs++;
			}
		}

		in.close();

		dataSet.attrName = new String[dataSet.numAttrs];
		dataSet.attrVals = new String[dataSet.numAttrs][];

		for (int i = 0; i < dataSet.numAttrs; i++) {
			String[] words = attr_list.get(i);
			dataSet.attrName[i] = words[0];
			if (words[1].equals("numeric")) {
				dataSet.attrVals[i] = null;
			} else {
				dataSet.attrVals[i] = new String[words.length - 1];
				for (int j = 1; j < words.length; j++) {
					dataSet.attrVals[i][j - 1] = words[j];
				}
			}
		}
	}

	public void readExamples(DataSet dataSet, String filestem) throws FileNotFoundException, IOException {
		String line;
		for (int traintest = 0; traintest < 2; traintest++) {
			ArrayList<int[]> ex_list = new ArrayList<int[]>();
			ArrayList<Integer> lab_list = new ArrayList<Integer>();

			if (traintest == 1)
				open_file(filestem + ".train");
			else
				try {
					open_file(filestem + ".test");
				} catch (FileNotFoundException e) {
					System.err.print("Continuing without test file...\n");
					dataSet.numTestExs = 0;
					dataSet.testEx = new int[0][];
					continue;
				}

			while ((line = read_line()) != null) {
				line = line.trim();
				if (line.equals(""))
					continue;

				String[] words = line.split("\\s+");
				if (words.length != dataSet.numAttrs + traintest) {
					String err = "wrong number of tokens at line " + line_count + " in file " + filename;
					System.err.println(err);
					throw new RuntimeException(err);
				}

				int ex[] = new int[dataSet.numAttrs];
				for (int i = 0; i < dataSet.numAttrs; i++) {
					if (dataSet.attrVals[i] == null) {
						try {
							ex[i] = Integer.parseInt(words[i]);
						} catch (NumberFormatException e) {
							System.err.println("Expected integer in field " + (i + 1) + " at line " + line_count
									+ " in file " + filename);
							throw e;
						}
					} else {
						int j;
						for (j = 0; j < dataSet.attrVals[i].length && !dataSet.attrVals[i][j].equals(words[i]); j++)
							;
						if (j >= dataSet.attrVals[i].length) {
							String err = "bad attribute value in field " + (i + 1) + " at line " + line_count
									+ " in file " + filename;
							System.err.println(err);
							throw new RuntimeException(err);
						}
						ex[i] = j;
					}
				}
				ex_list.add(ex);
				if (traintest == 1) {
					int lab;
					if (words[dataSet.numAttrs].equals(dataSet.className[0])) {
						lab = 0;
					} else if (words[dataSet.numAttrs].equals(dataSet.className[1])) {
						lab = 1;
					} else {
						String err = "unrecognized label at line " + line_count + " in file " + filename;
						System.err.println(err);
						throw new RuntimeException(err);
					}
					lab_list.add(new Integer(lab));
				}
			}

			if (traintest == 0) {
				dataSet.numTestExs = ex_list.size();
				dataSet.testEx = new int[0][];
				dataSet.testEx = (int[][]) ex_list.toArray(dataSet.testEx);
			} else {
				dataSet.numTrainExs = ex_list.size();
				dataSet.trainEx = new int[0][];
				dataSet.trainEx = (int[][]) ex_list.toArray(dataSet.trainEx);
				dataSet.trainLabel = new int[dataSet.numTrainExs];
				for (int i = 0; i < dataSet.numTrainExs; i++) {
					dataSet.trainLabel[i] = (lab_list.get(i)).intValue();
				}
			}
			in.close();
		}
	}


}
