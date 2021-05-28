
public class TestkNN {
	public static void main(String[] argv) throws Exception {
		if (argv.length < 5) {
            System.err.println("argument: filestem k iterMin iterMax increment numTrials");
            return;
        }
		
		// data set from filestem
        String fileStem = argv[0];
    	DataSetInput input = new FileInput(fileStem);
		DataSet d = new DiscreteDataSet(input);
		// number of data points to take for kNN
		int k = Integer.parseInt(argv[1]);
		// min and max number of iterations for weight training
		int iterMin = Integer.parseInt(argv[2]);
		int iterMax = Integer.parseInt(argv[3]);
		int increment = Integer.parseInt(argv[4]);
		// number of trials to be run per forest size
		int numTrials = Integer.parseInt(argv[5]);
		
		System.out.println("Data set contains " + d.numTrainExs + " examples.");
		System.out.println("Using " + k + " nearest points.");
		System.out.println("[numIters], [trialNum], [training error], [cross-set error]");
		for (int numIters = iterMin; numIters <= iterMax; numIters += increment) {
			// data set from filestem
			d = new DiscreteDataSet(input);
			double[][] error = TestHarness.computeError(d, numTrials, TestHarness.classifier.KNN, 0, false, k, numIters);
			for (int j = 0; j < numTrials; j++) {
				System.out.printf("%d, %d, %f, %f\n", numIters, j, error[j][0], error[j][1]);
			}
		}
	}
}
