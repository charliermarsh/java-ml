import java.util.Arrays;
import java.util.Comparator;

public class backwardsElimination extends StepwiseVariableSelection {
	public backwardsElimination(DataSet dataSet, Strategy strategy, boolean[] isEliminatedAttr,
			double[] instanceWeights) {
		this.dataSet = dataSet;
		this.strategy = strategy;
		this.isEliminatedAttr = isEliminatedAttr;
		this.instanceWeights = instanceWeights;
	}
	public void backwardsElimination() {
		
		double[][] distances = distanceSettingForBackward();
		
		int[][] orderedIndices = orderedIndices(distances);
		
		// calculate base error with no attribute elimination
		double baselineError = calcError(orderedIndices);
		
		distances = iterateEachAttribute(distances, orderedIndices, baselineError);
	}
	
	protected double[][] iterateEachAttribute(double[][] distances, int[][] orderedIndices, double baselineError) {
		int sum = 0;
		double[][] newdistance = new double[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
		int[][] orderedIndice = orderedIndices;
		double baselineerror = baselineError;
		// iterate over each attribute
		for (int m = 0; m < this.dataSet.numAttrs; m++) {
			double[][] temporaryDistances = new double[this.dataSet.numTrainExs][this.dataSet.numTrainExs];
			temporaryDistances = linearDistanceUpdate(distances, m);
			orderedIndice = computeNewNearestK(orderedIndice, temporaryDistances);
			double adjustedError = calcError(orderedIndice);

			// if error improved, keep attribute eliminated; else, retain
			boolean errorImproved = adjustedError < baselineerror;
			if ( errorImproved ) {
				this.isEliminatedAttr[m] = true;
				baselineerror = adjustedError;
				newdistance = temporaryDistances;
				sum++;
			}
			//if(m == this.d.numAttrs - 1)
			//	System.out.printf("%d attributes removed.\n", sum);
		}
		return newdistance;
	}
	
	protected double[][] distanceSettingForBackward() {
		double[][] distances = setCrossValidationDistance();
		for (int i = 0; i < distances.length; i++) {
			for (int j = i+1; j < distances.length; j++) {
				if (distances[i][j] == Double.POSITIVE_INFINITY) continue;
				
				distances[i][j] = getDistance(this.dataSet.trainEx[i], this.dataSet.trainEx[j]);
				distances[j][i] = distances[i][j];
			}
		}
		return distances;
	}
}
