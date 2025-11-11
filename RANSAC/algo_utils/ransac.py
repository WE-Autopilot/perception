
# When an initial estimate is provided
def RANSAC(data, initial_estimate, estimate_fn, test_fn, thresh, max_retry):

    # Evaluate the initial estimate and return if good enough
    bestEstimate = initial_estimate
    bestError = test_fn(data, initial_estimate)

    if bestError < thresh:
        return initial_estimate
    

    # Otherwise, proceed with RANSAC iterations
    for i in range(max_retry):
        estimate = estimate_fn(data)
        score = test_fn(data, estimate)

        if score < thresh:
            return estimate
    
        if score < bestError:
            bestEstimate = estimate
            bestError = score

    return bestEstimate

# Same thing, but without an initial estimate
def RANSAC(data, estimate_fn, test_fn, thresh, max_retry):

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