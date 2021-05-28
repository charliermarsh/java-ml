
public class Strategy {
	public Distance distanceStrategy;
	public Distance getDistanceStrategy() {
		return distanceStrategy;
	}

	public CrossValidation getCrossValidationStrategy() {
		return crossValidationStrategy;
	}

	public CrossValidation crossValidationStrategy;

	public Strategy(Distance distanceStrategy, CrossValidation crossValidationStrategy) {
		this.distanceStrategy = distanceStrategy;
		this.crossValidationStrategy = crossValidationStrategy;
	}
}