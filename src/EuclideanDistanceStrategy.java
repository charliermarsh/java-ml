public class EuclideanDistanceStrategy implements DistanceStrategy{
	@Override
	public double calcDistance(int pos1, int pos2) {
		double distance = Math.abs(pos1 - pos2);
		return distance;
	}
	public double calcPlusDistance(int pos1, int pos2) {
		double distance = Math.abs(pos1 + pos2);
		return distance;
	}
}