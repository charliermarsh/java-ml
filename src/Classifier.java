/**
 * This is the interface for a classifier.  A classifier only needs
 * three methods, one for evaluating examples, one for returning a
 * description of the learning algorithm used, and a third for
 * returning the "author" of the program.  Generally, the actual
 * learning will go into the constructer so that the computed
 * classifier is returned.
 */
public interface Classifier {

    /** A method for predicting the label of a given example <tt>ex</tt>
     * represented, as in the rest of the code, as an array of values
     * for each of the attributes.  The method should return a
     * prediction, i.e., 0 or 1.
     */
    public int predict(int[] ex);

    /** This method should return a very brief but understandable
     * description of the learning algorithm that is being used,
     * appropriate for posting on the class website.
     */
    public String algorithmDescription();

    /** This method should return the "author" of this program as you
     * would like it to appear on the class website.  You can use your
     * real name, or a pseudonym, or a name that identifies your
     * group.
    */
    public String author();
}
