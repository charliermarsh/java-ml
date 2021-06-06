
import java.util.Arrays;
import java.util.Comparator;
import java.util.PriorityQueue;

public class kNN implements Classifier {

	// data set of training examples
	private DataSet dataSet;
	// minimum possible value of k
	private final int kMin = 1;
	// maximum possible value of k
	private final int kMax = 5; // 15
	// cross-validated, optimized value of k
	private int kOpt = 7;
	// elimAttr[i] is true if attribute i has been eliminated
	private boolean[] isEliminatedAttr;
	// learning rate for weight training
	private final double learningRate = 0.05;
	// instanceWeights for training examples
	private double[] instanceWeights;
	// use 8 different sets for cross validation
	private int numSets = 8;
	private Strategy strategy;
	private StepwiseVariableSelection strate;

	public DataSet getDataSet() {
		return dataSet;
	}

	public int getkMin() {
		return kMin;
	}

	public int getkMax() {
		return kMax;
	}

	public int getkOpt() {
		return kOpt;
	}

	public boolean[] getIsEliminatedAttr() {
		return isEliminatedAttr;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public double[] getInstanceWeights() {
		return instanceWeights;
	}

	public Strategy getStrategy() {
		return strategy;
	}

	/**
	 * Constructor for the kNN machine learning algorithm. Takes as argument a data
	 * set. From then on, examples in the data set can be fed to predict() in return
	 * for classifications.
	 * 
	 * @return
	 */
	public kNN(DataSet dataSet, Strategy strategy) {
		/*
		 * Setup array labelledData so that it contains all the training data attributes
		 * along with that example's label.
		 */
		this.dataSet = dataSet;
		this.strategy = strategy;
		this.isEliminatedAttr = new boolean[this.dataSet.numAttrs];
		this.instanceWeights = new double[this.dataSet.numTrainExs];
		this.kOpt = findOptimalK(this.kMin, this.kMax);

		initInstanceWeights();
		backwardsElimination backwards = new backwardsElimination(dataSet, strategy, isEliminatedAttr, instanceWeights);
		backwards.backwardsElimination();
		traininstanceWeights(1);
	}

	/**
	 * Constructor for the kNN machine learning algorithm. Mainly used for testing
	 * the weight training heuristic.
	 */
	public kNN(DataSet dataSet, int kOpt, int numIteration, Strategy strategy) {
		/*
		 * Setup array labelledData so that it contains all the training data attributes
		 * along with that example's label.
		 */
		this.dataSet = dataSet;
		this.strategy = strategy;
		this.isEliminatedAttr = new boolean[this.dataSet.numAttrs];
		this.instanceWeights = new double[this.dataSet.numTrainExs];
		this.kOpt = kOpt;

		initInstanceWeights();
		traininstanceWeights(numIteration);
	}

	/**
	 * Constructor for the kNN machine learning algorithm. Takes as argument a data
	 * set and two indices, ignoring training examples between those indices to
	 * allow for cross validation. Additionally, takes an array of eliminated
	 * attributes, instanceWeights for each training example, and an optimal k
	 * value. From then on, examples in the data set can be fed to predict() in
	 * return for classifications. This constructor is used when a kNN instance is
	 * required for which no optimizations should be performed.
	 */
	public kNN(DataSet dataSet, int from, int to, int kOpt, boolean[] isEliminatedAttr, double[] instanceWeights,
			Strategy strategy) {
		/*
		 * Setup array labelledData so that it contains all the training data attributes
		 * along with that example's label.
		 */
		// create data set, excluding firstEx to lastEx examples
		DataSet subset = initSubset(dataSet, from, to);

		this.dataSet = subset;
		this.strategy = strategy;
		this.kOpt = kOpt;
		this.isEliminatedAttr = isEliminatedAttr;
		this.instanceWeights = instanceWeights;
	}

	private void initInstanceWeights() {
		for (int i = 0; i < this.instanceWeights.length; i++)
			this.instanceWeights[i] = 1.0;
	}

	private DataSet initSubset(DataSet dataSet, int from, int to) {
		DataSet subset = new DataSet();
		subset.numAttrs = dataSet.numAttrs;
		subset.numTrainExs = dataSet.numTrainExs - (to - from);
		subset.trainEx = new int[subset.numTrainExs][subset.numAttrs];
		subset.trainLabel = new int[subset.numTrainExs];

		initSubTrainData(dataSet, from, to, subset);
		return subset;
	}

	private void initSubTrainData(DataSet dataSet, int from, int to, DataSet subset) {
		for (int i = 0; i < from; i++) {
			subset.trainEx[i] = dataSet.trainEx[i];
			subset.trainLabel[i] = dataSet.trainLabel[i];
		}
		for (int i = to; i < dataSet.numTrainExs; i++) {
			subset.trainEx[i - to + from] = dataSet.trainEx[i];
			subset.trainLabel[i - to + from] = dataSet.trainLabel[i];
		}
	}

	/**
	 * Computes the squared distance between two integer vectors a and b.
	 */
	private double getDistance(int[] vector1, int[] vector2) {
		int len = Math.min(vector1.length, vector2.length);
		int distance = 0;
		for (int i = 0; i < len; i++) {
			// skip if attribute is eliminated
			if (this.isEliminatedAttr[i] == true)
				continue;
			distance += this.strategy.getDistanceStrategy().calcDistance(vector1[i], vector2[i]);
		}
		return distance;
	}

	/**
	 * Calculates the error over a labeled data set, returning a double that
	 * represents the percent error.
	 */
	private double calcErrorWithCrossValidation() {
		double error = 0.0;

		for (int setNum = 0; setNum < numSets; setNum++) {
			int from = setNum * this.dataSet.numTrainExs / numSets;
			int to = (setNum + 1) * this.dataSet.numTrainExs / numSets;

			// create new kNN using subset of data set
			kNN knn = new kNN(this.dataSet, from, to, this.kOpt, this.isEliminatedAttr, this.instanceWeights,
					this.strategy);

			for (int i = from; i < to; i++) {
				boolean isWrongPredict = knn.predict(this.dataSet.trainEx[i]) != this.dataSet.trainLabel[i];
				if (isWrongPredict)
					error++;
			}
		}
		return error;
	}

	/**
	 * Calculates the error over a labeled data set using pre-computed set of
	 * indices a in which a[i][j] represents index in training set d of the jth
	 * closest example to i. Returns a double that represents the percent error.
	 */
	private double calcError(int[][] kNNindices) {
		double error = 0.0;
		for (int i = 0; i < this.dataSet.numTrainExs; i++) {
			boolean isWrongPredict = voteCount(kNNindices[i]) != this.dataSet.trainLabel[i];
			if (isWrongPredict)
				error++;
		}
		return error;
	}

	/**
	 * Trains the instanceWeights of the attributes using backwards propagation on
	 * data set d, running T iterations.
	 */
	private void traininstanceWeights(int numIteration) {
		// get k nearest indices for each training example
		// as determined by cross validation

		int[][] kNNIndicesSet = kNearestWithCrossValidation();

		// run T iterations of weight training
		for (int t = 0; t < numIteration; t++) {

			// alter instanceWeights on each example
			for (int i = 0; i < this.dataSet.numTrainExs; i++) {

				// modify instanceWeights to satisfy example
				while (this.dataSet.trainLabel[i] != voteCount(kNNIndicesSet[i])) {
					for (int k = 0; k < this.kOpt; k++) {
						int neighborIndex = kNNIndicesSet[i][k];
						if (this.dataSet.trainLabel[neighborIndex] != this.dataSet.trainLabel[i])
							this.instanceWeights[neighborIndex] -= this.learningRate;
						else
							this.instanceWeights[neighborIndex] += this.learningRate;
					}
				}
			}
		}
		// System.out.println("instanceWeights trained.");
	}

	private int[][] kNearestWithCrossValidation() {
		int[][] kNNIndicesSet = new int[this.dataSet.numTrainExs][this.kOpt];

		for (int setNum = 0; setNum < numSets; setNum++) {
			int from = setNum * this.dataSet.numTrainExs / numSets;
			int to = (setNum + 1) * this.dataSet.numTrainExs / numSets;

			// create new kNN using subset of data set
			kNN knn = new kNN(this.dataSet, from, to, this.kOpt, this.isEliminatedAttr, this.instanceWeights,
					this.strategy);

			for (int i = from; i < to; i++)
				kNNIndicesSet[i] = knn.kNearest(this.kOpt, this.dataSet.trainEx[i]);
		}
		return kNNIndicesSet;
	}

	/**
	 * Uses backwards elimination to remove attributes from consideration that
	 * decrease the classifier's performance. To avoid recomputing distances,
	 * employs a linear-time distance update that just alters pre-computed distances
	 * based on attribute in question.
	 */

	/**
	 * Uses a cross-validation technique to find the optimal value of k for the kNN
	 * algorithm on data set d. Returns an integer k between k_min and k_max
	 * (inclusive).
	 */
	private int findOptimalK(int kMin, int kMax) {
		assert (kMax >= kMin);
		int[] kErrors = this.strategy.getCrossValidationStrategy().calcErrorByK(this, kMin, kMax);
		// set k to that of minimized error
		double minError = Double.MAX_VALUE;
		int kOpt = 0;
		for (int k = kMin; k <= kMax; k++) {
			if (kErrors[k - kMin] < minError) {
				minError = kErrors[k - kMin];
				kOpt = k;
			}
		}
		// System.out.printf("Optimal k chosen at k = %d\n", minK);
		return kOpt;
	}

	/**
	 * A class used to modularize comparisons for training examples based on a
	 * specific reference example ex. ex_index is used to avoid using the same
	 * training example as the reference example.
	 */
	public class exComparator implements Comparator {
		public boolean descending;
		private double[] dists;
		private int[] ex;
		private int exIndex = -1;
		public boolean isDescending = false;

		exComparator(int[] ex) {
			this.ex = ex;
		}

		/* Constructor which assumes base ex may be used in comparison */
		private exComparator(int[] ex, int exIndex) {
			this.ex = ex;
			this.exIndex = exIndex;
		}

		/* Constructor which allows for precomputed distances */
		exComparator(double[] dists, int exIndex) {
			this.dists = dists;
			this.exIndex = exIndex;
		}

		public int compare(Object object1, Object object2) {
			int trainExIndex1 = (int) (Integer) object1;
			int trainExIndex2 = (int) (Integer) object2;

			// ignore if training example is in data set
			if (trainExIndex1 == this.exIndex)
				return 1;
			if (trainExIndex2 == this.exIndex)
				return -1;

			// take column of min distance
			double dist1;
			double dist2;
			if (this.dists == null) {
				dist1 = getDistance(dataSet.trainEx[trainExIndex1], this.ex);
				dist2 = getDistance(dataSet.trainEx[trainExIndex2], this.ex);
			} else {
				dist1 = this.dists[trainExIndex1];
				dist2 = this.dists[trainExIndex2];
			}
			int result = 0;
			if (dist1 > dist2)
				result = -1;
			if (dist2 > dist1)
				result = 1;
			if (this.isDescending)
				result *= -1;
			return result;
		}
	}

	/**
	 * Calculates the indices of the k nearest training examples in data set d to
	 * example ex. Returns an array a in which a[i] is the index of the ith closest
	 * training example.
	 */
	public int[] kNearest(int k, int[] ex) {
		// indices of k best examples
		int[] kNNindices = new int[k];

		// record distances to avoid recalculation
		double[] dists = new double[this.dataSet.numTrainExs];

		// store indices in priority queue, sorted by distance to ex
		PriorityQueue<Integer> pq = new PriorityQueue<Integer>(k, new exComparator(ex));

		// search every example
		for (int i = 0; i < this.dataSet.numTrainExs; i++) {
			dists[i] = getDistance(this.dataSet.trainEx[i], ex);
			if (pq.size() >= k) {
				if (dists[i] < dists[pq.peek()]) {
					pq.remove();
					pq.add(i);
				}
			} else {
				pq.add(i);
			}
		}

		// pq returns worst index first; store backwards in indices
		for (int i = kNNindices.length - 1; i >= 0; i--)
			kNNindices[i] = pq.remove();
		return kNNindices;
	}

	/**
	 * Counts up the votes for the training examples with labels at indices listed
	 * in array a. Returns 0 or 1.
	 */
	private int voteCount(int[] indices) {
		double vote_1 = 0;
		double vote_0 = 0;
		int len = Math.min(indices.length, this.kOpt);
		for (int k = 0; k < len; k++) {
			int index = indices[k];
			if (this.dataSet.trainLabel[index] == 1)
				vote_1 += this.instanceWeights[index];
			else
				vote_0 += this.instanceWeights[index];
		}

		return (vote_1 > vote_0) ? 1 : 0;
	}

	/**
	 * A method for predicting the label of a given example <tt>ex</tt> represented,
	 * as in the rest of the code, as an array of values for each of the attributes.
	 * The method should return a prediction, i.e., 0 or 1.
	 */
	public int predict(int[] ex) {
		int[] kNNindices = kNearest(this.kOpt, ex);
		return voteCount(kNNindices);
	}

	/**
	 * This method should return a very brief but understandable description of the
	 * learning algorithm that is being used, appropriate for posting on the class
	 * website.
	 */
	public String algorithmDescription() {
		return "A kNN implementation with cross-validation, backwards elimination, and weight training.";
	}

	/**
	 * This method should return the "author" of this program as you would like it
	 * to appear on the class website. You can use your real name, or a pseudonym,
	 * or a name that identifies your group.
	 */
	public String author() {
		return "crm & dmrd";
	}

	/**
	 * A simple main for testing this algorithm. This main reads a filestem from the
	 * command line, runs the learning algorithm on this dataset, and prints the
	 * test predictions to filestem.testout.
	 * 
	 * @throws Exception
	 */
	public static void main(String argv[]) throws Exception {
		if (argv.length < 1) {
			System.err.println("argument: filestem");
			return;
		}

		String filestem = argv[0];

		DataSetInput input = new FileInput(filestem);
		DataSet d = new BinaryDataSet(input);

		Classifier c = new kNN(d, new Strategy(new EuclideanDistance(), new kFoldCrossValidation()));

		d.printTestPredictions(c, filestem);
	}

}
