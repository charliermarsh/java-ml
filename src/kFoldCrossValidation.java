public class kFoldCrossValidation implements CrossValidation{

	@Override
	public int[] calcErrorByK(kNN knn, int kMin, int kMax) {
		// use 8 different sets for cross validation
		int numSets = 8;
		int[] kErrors = new int[kMax - kMin + 1];
		
		for (int setNum = 0; setNum < numSets; setNum++) {
			int from = setNum*knn.getDataSet().numTrainExs/numSets;
			int to = (setNum+1)*knn.getDataSet().numTrainExs/numSets;
			
			// create new kNN using subset of data set
			kNN subKnn = new kNN(knn.getDataSet(), from, to, knn.getkOpt(),
					knn.getIsEliminatedAttr(), knn.getInstanceWeights(), knn.getStrategy());
			
			// test on held-out training examples
			for (int t = from; t < to; t++) {

				// get k_max best examples
				int[] kNNIndices = subKnn.kNearest(kMax, knn.getDataSet().trainEx[t]);

				// count votes by value of k
				double vote_0 = 0;
				double vote_1 = 0;
				for (int k = 0; k < kMax; k++) {
					int neighborIndex = kNNIndices[k];

					// track errors for appropriate k
					if (k >= kMin) {
						int predict = (vote_1 > vote_0)? 1 : 0;
						if (predict != knn.getDataSet().trainLabel[t]) 
							kErrors[k - kMin]++;
					}

					// continue to increment vote counts
					if (knn.getDataSet().trainLabel[neighborIndex] == 1)
						vote_1 += knn.getInstanceWeights()[neighborIndex];
					else
						vote_0 += knn.getInstanceWeights()[neighborIndex];
				}
				
				int predict = (vote_1 > vote_0)? 1 : 0;
				
				if (predict != knn.getDataSet().trainLabel[t]) 
					kErrors[kMax - kMin]++;
			}
		}
		return kErrors;
	}
}