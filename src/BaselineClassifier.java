
/**
 * This is the class for an extremely simple learning algorithm that
 * finds the most frequent class in the training data, and then
 * predicts that each new test example belongs to this class.
 */
public class BaselineClassifier implements Classifier {

    private int most_frequent_class;
    private String author = "Rob Schapire";
    private String description = "A very simple learning algorithm that, "
	+ "on each test example, predicts with the most frequent class seen "
	+ "during training";

    /**
     * This constructor takes as input a dataset and computes and
     * stores the most frequent class
     */
    public BaselineClassifier(DataSet d) {
	int count[] = new int[2];

	for (int i = 0; i < d.numTrainExs; i++)
	    count[d.trainLabel[i]]++;

	most_frequent_class = (count[1] > count[0] ? 1 : 0);
    }

    /** The prediction method ignores the given example and predicts
     * with the most frequent class seen during training.
     */
    public int predict(int[] ex) {
	return most_frequent_class;
    }

    /** This method returns a description of the learning algorithm. */
    public String algorithmDescription() {
	return description;
    }

    /** This method returns the author of this program. */
    public String author() {
	return author;
    }

    /** A simple main for testing this algorithm.  This main reads a
     * filestem from the command line, runs the learning algorithm on
     * this dataset, and prints the test predictions to filestem.testout.
     * @throws Exception 
     */
    public static void main(String argv[])
	throws Exception {

	if (argv.length < 1) {
	    System.err.println("argument: filestem");
	    return;
	}

	String filestem = argv[0];

	DataSetInput input = new FileInput(filestem);
	DataSet d = new DataSet(input);

	Classifier c = new BaselineClassifier(d);

        int correct = 0;
        for (int i = 0; i < d.numTrainExs; i++) {
            if (c.predict(d.trainEx[i]) == d.trainLabel[i])
                correct++;
        }

        System.out.println((100*correct/d.numTrainExs) + "%");

        d.printTestPredictions(c, filestem);
    }

}
