import hashlib
import statistics


class FlajoletMartin:
    def __init__(self, num_hashes=10):
        self.max_trailing_zeros = [0] * num_hashes

    def hash_function(self, item, seed):
        """Hash the given item using SHA-256 and return an integer."""
        # Combine element and seed into a string
        item_seeded = f"{seed}-{item}"  # like "0-element-1"
        # SHA-256
        hash_object = hashlib.sha256(item_seeded.encode())
        # Convert a hexadecimal hash value to an integer
        hash_value = int(hash_object.hexdigest(), 16)
        return hash_value

    def count_trailing_zeros(self, x):
        """Count the number of trailing zeros in the binary representation of a number."""

        #convert int to binary
        bin_x = bin(x)

        # remove the 0b prefix 
        slice_x = slice(2,None)
        bin_without0b = bin_x[slice_x]

        trailing_zeros_count = 0

        #count trailing zeros

        posoflast1 = bin_without0b.rfind('1')

        if posoflast1 == -1:
            return 0

        for i in range(posoflast1+1, len(bin_without0b)):
            trailing_zeros_count+=1

        return trailing_zeros_count

    def add(self, item):
        """Add a new item to the estimator and update the maximum trailing zeros."""

        for i in range(len(self.max_trailing_zeros)): # running through len of max_trailing_zeros, in our case 10
            hash_value= self.hash_function(item,i) # finding hash_value for example of (apple,0) 23501458907689157949873531067867675571870430681867888873248455490875782949626
            no_trailing = self.count_trailing_zeros(hash_value) # 2 
            self.max_trailing_zeros[i]= self.max_trailing_zeros[i] if self.max_trailing_zeros[i] > no_trailing else no_trailing

    def estimate_number(self):
        """Estimate the number of distinct elements by taking the median from multiple hashes."""
        # Calculate the median of the maximum number of trailing zeros of multiple hash functions
        median_trailing_zeros = statistics.median(self.max_trailing_zeros)
        # Estimate the cardinality from the median R
        estimated_number = 2 ** median_trailing_zeros / 0.75351
        return estimated_number, median_trailing_zeros


# Inspection Results
if __name__ == "__main__":
    fm = FlajoletMartin(num_hashes=20)
    unique_elements = [f"element-{i}" for i in range(100000)]  # Simulating 100,000 unique elements

    for element in unique_elements:
        fm.add(element)

    estimate, median_R = fm.estimate_number()
    print(f"Median R: {median_R}")
    print(f"Estimated Cardinality: {estimate}")