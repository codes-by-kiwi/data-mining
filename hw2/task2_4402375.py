from hashlib import sha256
import numpy as np
import functools

def create_hash_functions(num_hash_functions, size_bit_array):

    hash_functions = []

#partial reverse of the string,rest of the string is kept
    for i in range(num_hash_functions):
        create_hash = lambda acct, i=i: (int(sha256((acct[0:6][::-1] + acct[6:]).encode('utf-8')).hexdigest(), 16) + 2*i) % size_bit_array
        hash_functions.append(create_hash)
    return hash_functions

def add_to_bloom_filter(bloom_filter, hash_functions, bank_account):

    for func in hash_functions:
        pos = int(func(bank_account))
        bloom_filter[pos]= 1
    return bloom_filter

def check_bloom_filter(bloom_filter, hash_functions, bank_account): 
    dictofpos={}

    for func in hash_functions:
        pos = int(func(bank_account))
        if bank_account not in dictofpos:
            dictofpos[bank_account]=[]
        dictofpos[bank_account].append(pos)

    for pos in dictofpos[bank_account]:
            if bloom_filter[pos]==0:
                return False

    return True

if __name__ == "__main__":
    # This section can be used to debug your submission

    nr_bank_accounts = 100_000

    # Create a list of legal bank account numbers
    real_bank_accounts = ["real" + str(i) for i in range(nr_bank_accounts)]

    # Set up the Bloom filter as an array 8 times as big as the number of bank accounts
    bloom_filter = [0] * 8*nr_bank_accounts
    # Experiment with 2 hash functions (try raising it to 30)
    hash_functions = create_hash_functions(2, 8*nr_bank_accounts)
    # Enter all valid account numbers
    for account in real_bank_accounts:
        add_to_bloom_filter(bloom_filter, hash_functions, account)

    # Calulate the false positive rate
    fake_bank_accounts = ["fake" + str(i) for i in range(nr_bank_accounts)]
    false_positives = 0
    for fake_account in fake_bank_accounts:
        if check_bloom_filter(bloom_filter, hash_functions, fake_account):
            false_positives += 1 
    print(f"False positive rate: {false_positives/nr_bank_accounts}")

    print("Fraction of bits set: ", np.sum(bloom_filter)/(nr_bank_accounts*8))
        
    print("Is real12345 a valid account number?", check_bloom_filter(bloom_filter, hash_functions, "real12345"))
    print("Is real123456 a valid account number?", check_bloom_filter(bloom_filter, hash_functions, "real123456"))
    print("Is 12345 a valid account number?", check_bloom_filter(bloom_filter, hash_functions, "12345"))
