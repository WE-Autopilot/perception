import numpy as np

class RANSAC:
    # When an initial estimate is provided
    def RANSAC_fn(self, data: np.ndarray, initial_estimate, estimate_fn, test_fn, max_retry=10, thresh=5):

        # Evaluate the initial estimate and return if good enough
        bestEstimate = initial_estimate
        bestError = test_fn(data, initial_estimate)
        if bestError < thresh:
            return initial_estimate

        # Otherwise, proceed with RANSAC iterations
        for i in range(max_retry):
            print(i)
            estimate = estimate_fn(data)
            score = test_fn(data, estimate)

            if score < thresh:
                return estimate
            if score < bestError:
                bestEstimate = estimate
                bestError = score
            
        return bestEstimate

    # When no initial estimate is provided
    def RANSAC_noInit(self, data, estimate_fn, test_fn, thresh, max_retry):

        bestEstimate = None
        bestError = float('inf')

        for i in range(max_retry):
            estimate = estimate_fn(data)
            score = test_fn(data, estimate)

            if score < thresh:
                return estimate
            if score < bestError:
                bestEstimate = estimate
                bestError = score
        
        return bestEstimate