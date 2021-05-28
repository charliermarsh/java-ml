import java.util.Arrays;
import java.util.Comparator;


public abstract class StepwiseVariableSelection {
	public Strategy strategy;
	public DataSet dataSet;
	public boolean[] isEliminatedAttr;
	public int numSets = 8;
	public double[] instanceWeights;
	public int kOpt = 7;
	
	protected double calcError(int[][] kNNindices) {
		double error = 0.0;
		for (int i = 0; i < this.dataSet.numTrainExs; i++) {
			boolean isWrongPredict = voteCount(kNNindices[i]) != this.dataSet.trainLabel[i];
			if (isWrongPredict)
				error++;
		}
		return error;
	}
	protected int voteCount(int[] indices) {
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
    	
    	return (vote_1 > vote_0)? 1 : 0;
	}
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
	    public exComparator(int[] ex, int exIndex) {
		  this.ex = ex;
		  this.exIndex = exIndex;
		}
	    
	    /* Constructor which allows for precomputed distances */
	    protected exComparator(double[] dists, int exIndex) {
			this.dists = dists;
			this.exIndex = exIndex;
		}

	    public int compare(Object object1, Object object2) {
	    	int trainExIndex1 = (int)(Integer)object1;
	    	int trainExIndex2 = (int)(Integer)object2;
	    	
	    	// ignore if training example is in data set
	    	if (trainExIndex1 == this.exIndex) return 1;
	    	if (trainExIndex2 == this.exIndex) return -1;
	        
	        // take column of min distance
	    	double dist1;
	    	double dist2;
	    	if (this.dists == null) {
	    	    dist1 = getDistance(dataSet.trainEx[trainExIndex1], this.ex);
	    		dist2 = getDistance(dataSet.trainEx[trainExIndex2], this.ex);
	    	}
	    	else {
	    		dist1 = this.dists[trainExIndex1];
	    		dist2 = this.dists[trainExIndex2];	    		
	    	}
	    	int result = 0;
	    	if (dist1 > dist2) result = -1;
    		if (dist2 > dist1) result = 1;
    		if (this.isDescending) result *= -1;
    		return result;
	    }
	}
	protected int[][] orderedIndices(double[][] distances) {
		// orderedIndices[i][j] is index of jth closest example to i
		int[][] orderedIndices = new int[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
		for (int i = 0; i < orderedIndices.length; i++) {
			// annoying Integer to int casting issues
			Integer[] alpha = new Integer[this.dataSet.numTrainExs];
			for (int j = 0; j < orderedIndices.length; j++) {
				alpha[j] = j;
			}
			exComparator comparator = new exComparator(distances[i], i);
			comparator.descending = true;
			Arrays.sort(alpha, comparator);
			for (int j = 0; j < orderedIndices.length; j++) {
				orderedIndices[i][j] = alpha[j];
			}
		}
		return orderedIndices;
	}
	
	protected double getDistance(int[] vector1, int[] vector2) {
		int len = Math.min(vector1.length, vector2.length);
		int distance = 0;
		for (int i = 0; i < len; i++) {
			// skip if attribute is eliminated
			if (this.isEliminatedAttr[i] == true) continue;
			distance += this.strategy.getDistanceStrategy().calcDistance(vector1[i], vector2[i]);
		}
        return distance;
    }
	protected int[][] computeNewNearestK(int[][] orderedIndice, double[][] temporaryDistances) {
		// compute new k nearest
		int[][] orderedIndices = new int[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
		for (int i = 0; i < this.dataSet.numTrainExs; i++) {
			// annoying Integer to int casting issues
			Integer[] integers = new Integer[this.dataSet.numTrainExs];
			for (int j = 0; j < orderedIndice.length; j++) {
				integers[j] = orderedIndice[i][j];
			}
			exComparator comparator = new exComparator(temporaryDistances[i], i);
			comparator.descending = true;
			Arrays.sort(integers, comparator);
			for (int j = 0; j < orderedIndice.length; j++) {
				orderedIndices[i][j] = integers[j];
			}
		}
		return orderedIndices;
	}
	protected double[][] linearDistanceUpdate(double[][] distances, int m) {
		// linear-time distance update
		double[][] temporaryDistances = new double[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
		for (int i = 0; i < distances.length; i++) {
			for (int j = i+1; j < distances.length; j++) {
				double distanceCalculation = strategy.getDistanceStrategy().calcDistance(this.dataSet.trainEx[i][m], this.dataSet.trainEx[j][m]);
				temporaryDistances[i][j] = distances[i][j] - distanceCalculation;
				temporaryDistances[j][i] = temporaryDistances[i][j];
			}
		}
		return temporaryDistances;
	}
	protected double[][] setCrossValidationDistance() {
		// calculate all distances to avoid recomputation
		double[][] distances = new double[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
		
		for (int setNum = 0; setNum < numSets; setNum++) {
			int from = setNum*this.dataSet.numTrainExs/numSets;
			int to = (setNum+1)*this.dataSet.numTrainExs/numSets;

			for (int t = from; t < to; t++) {
				for (int s = t+1; s < to; s++) {
					distances[t][s] = Double.POSITIVE_INFINITY;
					distances[s][t] = distances[t][s];
				}
			}
		}
		return distances;
	}
}
