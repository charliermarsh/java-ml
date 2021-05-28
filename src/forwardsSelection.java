import java.util.Arrays;
import java.util.Comparator;

public class forwardsSelection extends StepwiseVariableSelection {
	
	public void forwardsSelection() {
		removeAllAttributes();
		
		double[][] distances = setCrossValidationDistance();
		int[][] orderedIndices = orderedIndices(distances);
		
		// calculate base error with no attribute elimination
		double baselineError = calcError(orderedIndices);
		boolean attributeAdded;
		
		do {
			attributeAdded = false;
			double minError = Double.POSITIVE_INFINITY;
			int minErrorIndex = -1;
			double[][] minErrorDistances = new double[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
			
			// iterate over each attribute
			for (int m = 0; m < this.dataSet.numAttrs; m++) {
				boolean mthEliminateAttributNotEmpty = !this.isEliminatedAttr[m];
				if (mthEliminateAttributNotEmpty) continue;
				
				double[][] newDistances = new double[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
				newDistances = linearForwardDistanceUpdate(distances, m);
				
				computeNewNearestK(orderedIndices, newDistances);

				double adjustedError = calcError(orderedIndices);

				// if error improved, keep attribute eliminated; else, retain
				boolean errorImproved = ( adjustedError < minError );
				if ( errorImproved ) {
					minError = adjustedError;
					minErrorIndex = m;
					minErrorDistances = newDistances;
				}
			}
			
			boolean baselineErrorImproved = (minError < baselineError);
			
			if ( baselineErrorImproved ) {
				this.isEliminatedAttr[minErrorIndex] = false;
				distances = minErrorDistances;
				attributeAdded = true;
				//System.out.println("Added attribute " + minErrorIndex);
			}
		} while (attributeAdded);
	}
	public forwardsSelection(DataSet dataSet, Strategy strategy, boolean[] isEliminatedAttr,
			double[] instanceWeights) {
		this.dataSet = dataSet;
		this.strategy = strategy;
		this.isEliminatedAttr = isEliminatedAttr;
		this.instanceWeights = instanceWeights;
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

	
	protected double[][] linearForwardDistanceUpdate(double[][] distances, int m) {
		// linear-time distance update
		double[][] temporaryDistances = new double[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
		for (int i = 0; i < distances.length; i++) {
			for (int j = i+1; j < distances.length; j++) {
				double distanceCalculation = strategy.distanceStrategy.calcPlusDistance(this.dataSet.trainEx[i][m], this.dataSet.trainEx[j][m]);
				temporaryDistances[i][j] = distances[i][j] - distanceCalculation;
				temporaryDistances[j][i] = temporaryDistances[i][j];
			}
		}
		
		return temporaryDistances;
	}
	protected void removeAllAttributes() {
		// remove all attributes
		for (int i = 0; i < this.isEliminatedAttr.length; i++)
			this.isEliminatedAttr[i] = true;
	}

	
	
}