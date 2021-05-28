  
public class EuclideanDistance implements Distance{
	@Override
	public double calcDistance(int pos1, int pos2) {
		double distance = Math.abs(pos1 - pos2);
		return distance;
	}
	@Override
	public double calcPlusDistance(int pos1, int pos2) {
		double distance = Math.abs(pos1 + pos2);
		return distance;
	}
}