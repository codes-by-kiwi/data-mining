import random
probabilities={}

def mock_datastream():
    for _ in range(10_000):
        yield random.gauss(10, 100) * (1 + 0.0005)

def reservoir_sampling(k, datastream):

    sample = [] 
    count_dp = 0 #count of data points

    for datapoint in datastream():
        count_dp += 1

    if k > count_dp:
        print("Error: Reservoir Sampling - You cannot sample more than existing amount of datapoints")

    for index, transaction in enumerate(datastream()):
        if index < k:
            sample.append(transaction)
        else:
            prob_idx=k/(index+1)
            if random.uniform(0,1) < prob_idx: # making sure that prob_idx is greater than a random decimal generated between 0 and 1
                j = random.randint(0,k-1) # making sure that j is a number between 0 and reservoir size
                sample[j] = transaction

    return sample

if __name__ == "__main__":
    sample = reservoir_sampling(5000, mock_datastream)
    print(len(sample))
    print(sample)