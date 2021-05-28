
public class Sigmoid implements Activation{
	public double getActivation(double d) {
		return 1.0/(1.0 + Math.exp(-d));
		
	}
	public double getDerivation(double d) {
		double derivation = this.getActivation(d);
		return derivation * (1.0 - derivation);	
	}
}
