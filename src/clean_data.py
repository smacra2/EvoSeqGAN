import pickle
import numpy as np


# Function to read a file and match specified sequences within it
# Reference sequences start with tag1, candidate sequences start with tag2
# Ignores sequences containing more than 50% gaps
def read_matches(fname, tag1, tag2, cut_length=100):
    # Read in all lines from input file
    f = open(fname)
    txt = f.readlines()
    f.close()

    ref = []  # for reference sequences
    refind = []  # for indices where reference sequences are found in file
    cnd = []  # likewise for candidate sequences
    cndind = []

    # Find lines containing tag1
    for i in range(0, len(txt)):
        tmp = txt[i]
        if tmp[0:len(tag1)] == tag1:  # if tag1 found on line
            # add sequence into reference list after conversion to all uppercase
            x = tmp.split()
            ref.append(x[-1].upper())
            # store sequence line number also
            refind.append(i)

    # Find lines containing tag2
    for i in range(0, len(txt)):
        tmp = txt[i]
        if tmp[0:len(tag2)] == tag2:  # if tag2 found on line
            # add sequence into candidate list after conversion to all uppercase
            x = tmp.split()
            cnd.append(x[-1].upper())
            # store sequence line number also
            cndind.append(i)

    # Handle case where reference sequence exists, but not the corresponding candidate sequence
    # Match with read sequence
    concat_ref = []
    concat_cnd = []
    refind.append(len(txt))

    i = 0
    j = 0

    processed_count = 0
    # Generally expect every reference sequence to be followed by a candidate sequence
    # If two reference sequences appear with no candidate sequence in between, reject read sequence
    while i < len(ref) and j < len(cnd):
        if cndind[j] > refind[i] and cndind[j] < refind[i + 1]:
            concat_ref.append(ref[i])
            concat_cnd.append(cnd[j])
            processed_count += 1
            i += 1
            j += 1
        elif cndind[j] < refind[i]:
            j += 1
        else:
            i += 1

    print("Number of processed pairs: " + str(processed_count))

    temp_ref = ''
    for seq in concat_ref:
        temp_ref += str(seq)

    temp_cnd = ''
    for seq in concat_cnd:
        temp_cnd += str(seq)

    temp_ref = list(temp_ref)
    temp_cnd = list(temp_cnd)
    print("Length of temp_ref and temp_cnd before removing double gaps: %d %d" % (len(temp_ref), len(temp_cnd)))

    # Remove gaps that occur at same position in both ref and cnd sequences
    for i in range(min(len(temp_ref), len(temp_cnd)) - 1, -1, -1):
        if temp_ref[i] == '-' and temp_cnd[i] == '-':
            temp_ref[i] = "REPLACE"
            temp_cnd[i] = "REPLACE"

    temp_ref = "".join(temp_ref)
    temp_ref = temp_ref.replace("REPLACE", "")
    temp_cnd = "".join(temp_cnd)
    temp_cnd = temp_cnd.replace("REPLACE", "")
    print("Length of temp_ref and temp_cnd after removing double gaps: %d %d" % (len(temp_ref), len(temp_cnd)))

    split = cut_length  # cut every 100 pieces to account for variable length of initial data
    match_ref = [temp_ref[i:i+split] for i in range(0, len(temp_ref), split)]
    match_cnd = [temp_cnd[i:i+split] for i in range(0, len(temp_cnd), split)]

    # Before appending to output list, remove alignments that are composed of more than 50% gaps
    matches = []
    removed_count = 0
    for i in range(0, min(len(match_ref), len(match_cnd))):
        temp_ref = match_ref[i]
        temp_cnd = match_cnd[i]

        num_gap_ref = 0
        num_gap_cnd = 0
        for char in temp_ref:
            if char == '-':
                num_gap_ref += 1
        for char in temp_cnd:
            if char == '-':
                num_gap_cnd += 1

        if num_gap_ref > (cut_length / 2) or num_gap_cnd > (cut_length / 2):
            removed_count += 1
        else:
            matches.append([match_ref[i], match_cnd[i]])

    print("Number of alignments removed for having >50%% gaps: %d" % removed_count)

    # Return the list of matched sequences
    return matches


# Function to read a file and match specified sequences within it
# Reference sequences start with tag1, candidate sequences start with tag2
# *** Same as read_matches function but specifically to find sequence chunks with no gaps ***
def read_matches_no_gaps(fname, tag1, tag2, cut_length=100):
    # Read in all lines from input file
    f = open(fname)
    txt = f.readlines()
    f.close()

    ref = []  # for reference sequences
    refind = []  # for indices where reference sequences are found in file
    cnd = []  # likewise for candidate sequences
    cndind = []

    # Find lines containing tag1
    for i in range(0, len(txt)):
        tmp = txt[i]
        if tmp[0:len(tag1)] == tag1:  # if tag1 found on line
            # add sequence into reference list after conversion to all uppercase
            x = tmp.split()
            ref.append(x[-1].upper())
            # store sequence line number also
            refind.append(i)

    # Find lines containing tag2
    for i in range(0, len(txt)):
        tmp = txt[i]
        if tmp[0:len(tag2)] == tag2:  # if tag2 found on line
            # add sequence into candidate list after conversion to all uppercase
            x = tmp.split()
            cnd.append(x[-1].upper())
            # store sequence line number also
            cndind.append(i)

    # Handle case where reference sequence exists, but not the corresponding candidate sequence
    # Match with read sequence
    concat_ref = []
    concat_cnd = []
    refind.append(len(txt))

    i = 0
    j = 0

    processed_count = 0
    # Generally expect every reference sequence to be followed by a candidate sequence
    # If two reference sequences appear with no candidate sequence in between, reject read sequence
    while i < len(ref) and j < len(cnd):
        if cndind[j] > refind[i] and cndind[j] < refind[i + 1]:
            concat_ref.append(ref[i])
            concat_cnd.append(cnd[j])
            processed_count += 1
            i += 1
            j += 1
        elif cndind[j] < refind[i]:
            j += 1
        else:
            i += 1

    print("Number of processed pairs: " + str(processed_count))

    temp_ref = ''
    for seq in concat_ref:
        temp_ref += str(seq)

    temp_cnd = ''
    for seq in concat_cnd:
        temp_cnd += str(seq)

    temp_ref = list(temp_ref)
    temp_cnd = list(temp_cnd)
    print("Length of temp_ref and temp_cnd before removing double gaps: %d %d" % (len(temp_ref), len(temp_cnd)))

    # Remove positions with gaps that occur at either or both positions in ref and cnd sequences
    for i in range(min(len(temp_ref), len(temp_cnd)) - 1, -1, -1):
        if temp_ref[i] == '-' or temp_cnd[i] == '-':
            temp_ref[i] = "REPLACE"
            temp_cnd[i] = "REPLACE"

    temp_ref = "".join(temp_ref)
    temp_ref = temp_ref.replace("REPLACE", "")
    temp_cnd = "".join(temp_cnd)
    temp_cnd = temp_cnd.replace("REPLACE", "")
    print("Length of temp_ref and temp_cnd after removing gap positions: %d %d" % (len(temp_ref), len(temp_cnd)))

    split = cut_length  # cut every cut_length pieces to account for variable length of initial data
    match_ref = [temp_ref[i:i+split] for i in range(0, len(temp_ref), split)]
    match_cnd = [temp_cnd[i:i+split] for i in range(0, len(temp_cnd), split)]

    # Append to output list
    matches = []
    for i in range(0, min(len(match_ref), len(match_cnd))):
        matches.append([match_ref[i], match_cnd[i]])

    # Return the list of matched sequences
    return matches


# Function to one-hot encode the sequences in matches list and label with yhat
# A: 10000, C: 01000, G: 00100, T: 00010, -: 00001
def prepare_data(matches, yhat):
    X = []
    Y = []

    # For each matched sequence in matches list
    for i in range(0, len(matches)):
        ref = matches[i][0]
        cnd = matches[i][1]
        # Initialize encoding sequence
        tmp = np.zeros((1, len(ref), 10))
        # Encode reference sequence characters
        for j in range(0, len(ref)):
            if ref[j] == 'A':
                tmp[0, j, 0] = 1
            elif ref[j] == 'C':
                tmp[0, j, 1] = 1
            elif ref[j] == 'G':
                tmp[0, j, 2] = 1
            elif ref[j] == 'T':
                tmp[0, j, 3] = 1
            else:
                tmp[0, j, 4] = 1
            # Encode candidate sequence characters
            if cnd[j] == 'A':
                tmp[0, j, 5] = 1
            elif cnd[j] == 'C':
                tmp[0, j, 6] = 1
            elif cnd[j] == 'G':
                tmp[0, j, 7] = 1
            elif cnd[j] == 'T':
                tmp[0, j, 8] = 1
            else:
                tmp[0, j, 9] = 1

        # Add encoded sequences together with provided label
        X.append(tmp)
        Y.append([yhat])

    return X, Y


# Function to prepare only chunks of X length of real data
# Not used for GAN training, see other clean_data.py file for how to clean indels
def real_data(chunk_size=500):
    realMatches = read_matches('Real_Alignments_48.000.000_Lines.txt', 's hg38', 's _HPGPNRMPC ', chunk_size)
    print("Number of real paired sequences: " + str(len(realMatches)))

    X_real, Y_real = prepare_data(realMatches, 1)

    X_train = []
    Y_train = []
    reserved_test_set = len(realMatches) // 10
    size_train_set = len(realMatches) - reserved_test_set
    for i in range(0, size_train_set):
        X_train.append(X_real[i])
        Y_train.append(Y_real[i])

    print("Size X_train: " + str(len(X_train)))
    print("Size Y_train: " + str(len(Y_train)))

    pickle_out = open('realData_' + str(chunk_size), "wb")
    pickle.dump(X_train, pickle_out)
    pickle.dump(Y_train, pickle_out)
    pickle_out.close()


# Function to prepare only chunks of X length of real data containing no gaps
# Primary function used for GAN training of substitution only model
def real_data_no_gaps(chunk_size=500):
    realMatches = read_matches_no_gaps('Real_Alignments_48.000.000_Lines.txt', 's hg38', 's _HPGPNRMPC ', chunk_size)
    print("Number of real paired sequences (with no gaps): " + str(len(realMatches)))

    X_real, Y_real = prepare_data(realMatches, 1)

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    reserved_test_set = len(realMatches) // 10
    size_train_set = len(realMatches) - reserved_test_set
    for i in range(0, size_train_set):
        X_train.append(X_real[i])
        Y_train.append(Y_real[i])

    print("Size X_train: " + str(len(X_train)))
    print("Size Y_train: " + str(len(Y_train)))

    pickle_out = open('realData_gapless_' + str(chunk_size), "wb")
    pickle.dump(X_train, pickle_out)
    pickle.dump(Y_train, pickle_out)
    pickle_out.close()

    for i in range(size_train_set, len(realMatches)):
        X_test.append(X_real[i])
        Y_test.append(Y_real[i])

    print("Size X_test: " + str(len(X_test)))
    print("Size Y_test: " + str(len(Y_test)))

    pickle_out = open('testData_gapless_' + str(chunk_size), "wb")
    pickle.dump(X_test, pickle_out)
    pickle.dump(Y_test, pickle_out)
    pickle_out.close()


# Main function for creating real and fake train and test sequences, not used for actual GAN training
def main():
    # Read from specified file sequences starting with tag1 and tag2
    realMatches = read_matches('Real_Alignments_32.000.000_Lines.txt', 's hg38', 's _HPGPNRMPC ', 100)
    fakeMatches = read_matches('Fake_Alignments_140.000_Lines.txt', 's Human.', 's _HR ', 100)

    print("Number of real paired sequences: " + str(len(realMatches)))
    print("Number of fake paired sequences: " + str(len(fakeMatches)))

    # Encode the sequence characters with binary label: 1 for real, 0 for fake
    X_real, Y_real = prepare_data(realMatches, 1)
    X_fake, Y_fake = prepare_data(fakeMatches, 0)

    # Reorder the fake and real sequences randomly to prepare for discriminator
    np.random.seed(999)
    p = np.random.permutation(len(realMatches) + len(fakeMatches))

    X_train = []
    Y_train = []
    reserved_test_set = 4000
    size_train_set = len(realMatches) + len(fakeMatches) - reserved_test_set

    for i in range(0, size_train_set):
        if p[i] < len(realMatches):
            X_train.append(X_real[p[i]])
            Y_train.append(Y_real[p[i]])
        else:
            X_train.append(X_fake[p[i] - len(realMatches)])
            Y_train.append(Y_fake[p[i] - len(realMatches)])

    print("Size X_train: " + str(len(X_train)))
    print("Size Y_train: " + str(len(Y_train)))

    # Write matched data and reordered data into a pickle file
    pickle_out = open('cleanData', "wb")
    pickle.dump(realMatches, pickle_out)
    pickle.dump(fakeMatches, pickle_out)
    pickle.dump(X_train, pickle_out)
    pickle.dump(Y_train, pickle_out)
    pickle_out.close()

    # Repeat for held-out test set
    X_test = []
    Y_test = []

    for i in range(size_train_set, size_train_set + reserved_test_set):
        if p[i] < len(realMatches):
            X_test.append(X_real[p[i]])
            Y_test.append(Y_real[p[i]])
        else:
            X_test.append(X_fake[p[i] - len(realMatches)])
            Y_test.append(Y_fake[p[i] - len(realMatches)])

    print("Size X_test: " + str(len(X_test)))
    print("Size Y_test: " + str(len(Y_test)))

    # Write matched data and reordered data into a pickle file
    pickle_out = open('testData', "wb")
    pickle.dump(X_test, pickle_out)
    pickle.dump(Y_test, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    # main()
    # real_data(5000)
    real_data_no_gaps(15000)
