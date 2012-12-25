/*
 * A basic test harness for the different algorithms
 */
import java.util.*;
import java.io.*;
public class TestHarness {

    private static int FOLDNUM = 1;
    private static int FOLDDENOM = 8;
    private static int numTrees = 100;

    public enum classifier {
        DT, DF, SLNN, MLNN, KNN, BASE
    };

    static classifier algo;

    public static void runTrials(DataSet d, int numTrials) {
        Random random = new Random();
        int crossSize = FOLDNUM * d.numTrainExs / FOLDDENOM;
        int[][] oEx = new int[d.numTrainExs][];
        int[] oLabel = new int[d.numTrainExs];
        for (int i = 0; i < d.numTrainExs; i++) {
            oEx[i] = d.trainEx[i];
            oLabel[i] = d.trainLabel[i];
        }

        d.numTrainExs -= crossSize;
        d.trainEx = new int[d.numTrainExs][];
        d.trainLabel = new int[d.numTrainExs];

        System.out.println("Training classifier on " + d.numTrainExs
                + " examples with " + numTrials + " trials.  Testing on "
                + crossSize + " examples");
        int totalCorrect = 0;
        for (int trial = 0; trial < numTrials; trial++) {

            /*Shuffle the dataset to get a training/test set for each trial*/
            for (int i = 0; i < oEx.length; i++) {
                int swap = random.nextInt(oEx.length - i);
                int[] tempEx = oEx[swap];
                oEx[swap] = oEx[oEx.length - i - 1];
                oEx[oEx.length - i - 1] = tempEx;

                /*Same for labels*/
                int tempLabel = oLabel[swap];
                oLabel[swap] = oLabel[oEx.length - i - 1];
                oLabel[oEx.length - i - 1] = tempLabel;
            }

            for (int i = 0; i < d.numTrainExs; i++) {
                d.trainEx[i] = oEx[i];
                d.trainLabel[i] = oLabel[i];
            }

            Classifier c;
            switch (algo) {
                case DT:
                    c = new DecisionTree(d);
                    break;
                case DF:
                    c = new DecisionForest(d, numTrees);
                    break;
                case KNN:
                    c = new kNN(d);
                    break;
                case SLNN:
                    c = new SingleLayerNeuralNet(d);
                    break;
                case MLNN:
                    c = new MultiLayerNeuralNet(d);
                    break;
                default:
                    c = new BaselineClassifier(d);
            }


            int correct = 0;
            for (int ex = oEx.length - crossSize; ex < oEx.length; ex++) {
                if (c.predict(oEx[ex]) == oLabel[ex])
                    correct++;
            }
            
            totalCorrect += correct;
            System.out.println("Performance on cross set: "
                    + (100*correct / crossSize) + "%");

        }
        System.out.println("Average percent correct: "
                    + (100*totalCorrect / (crossSize * numTrials))  + "%");
        return;
    }

    /*
     * Simple main for testing.
     */
    public static void main(String argv[])
        throws FileNotFoundException, IOException {

        if (argv.length < 3) {
            System.err.println("argument: filestem classifier #runs classifierArgs");
            System.err.println("Classifier options: dt, df, knn, slnn, mlnn");
            return;
        }

        DataSet d;
        switch (argv[1]) {
            case "dt":
                System.out.println("Using decision tree");
                algo = classifier.DT;
                d = new DiscreteDataSet(argv[0]);
                break;
            case "df":
                System.out.println("Using decision forest");
                if (argv.length == 4) { numTrees = Integer.parseInt(argv[3]); }
                algo = classifier.DF;
                d = new DiscreteDataSet(argv[0]);
                break;
            case "knn":
                System.out.println("Using k-nearest-neighbor");
                algo = classifier.KNN;
                d = new NumericDataSet(argv[0]);
                break;
            case "slnn":
                System.out.println("Using single layer neural net");
                algo = classifier.SLNN;
                d = new BinaryDataSet(argv[0]);
                break;
            case "mlnn":
                System.out.println("Using multilayer neural net");
                algo = classifier.MLNN;
                d = new BinaryDataSet(argv[0]);
                break;
            default:
                System.out.println("Using baseline classifier");
                algo = classifier.BASE;
                d = new DataSet(argv[0]);
        }

        runTrials(d, Integer.parseInt(argv[2]));
    }
}
