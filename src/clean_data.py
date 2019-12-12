import pickle
import numpy as np


# Function to read a file and match specified sequences within it
# Reference sequences start with tag1, candidate sequences start with tag2
def read_matches(fname, tag1, tag2):
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

    count = 0

    # Generally expect every reference sequence to be followed by a candidate sequence
    # If two reference sequences appear with no candidate sequence in between, reject read sequence
    while i < len(ref) and j < len(cnd):
        if cndind[j] > refind[i] and cndind[j] < refind[i + 1]:
            concat_ref.append(ref[i])
            concat_cnd.append(cnd[j])
            count += 1
            i += 1
            j += 1
        elif cndind[j] < refind[i]:
            j += 1
        else:
            i += 1

    print("Number of processed pairs: " + str(count))

    temp_ref = ''
    for seq in concat_ref:
        temp_ref += str(seq)

    temp_cnd = ''
    for seq in concat_cnd:
        temp_cnd += str(seq)

    split = 100  # cut every 100 pieces to account for variable length of initial data
    match_ref = [temp_ref[i:i+split] for i in range(0, len(temp_ref), split)]
    match_cnd = [temp_cnd[i:i+split] for i in range(0, len(temp_cnd), split)]

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


if __name__ == "__main__":

    # Read from specified file sequences starting with tag1 and tag2
    realMatches = read_matches('Real_Alignments_16.000.000_Lines.txt', 's hg38', 's _HPGPNRMPC ')
    fakeMatches = read_matches('Fake_Alignments_100.000_Lines.txt', 's Human.', 's _HR ')

    print("Number of real paired sequences: " + str(len(realMatches)))
    print("Number of fake paired sequences: " + str(len(fakeMatches)))

    # Encode the sequence characters with binary label: 1 for real, 0 for fake
    X_real, Y_real = prepare_data(realMatches, 1)
    X_fake, Y_fake = prepare_data(fakeMatches, 0)

    # Reorder the fake and real sequences randomly to prepare for discriminator
    p = np.random.permutation(len(realMatches) + len(fakeMatches))

    X = []
    Y = []

    for i in range(0, len(realMatches) + len(fakeMatches)):
        if p[i] < len(realMatches):
            X.append(X_real[p[i]])
            Y.append(Y_real[p[i]])
        else:
            X.append(X_fake[p[i] - len(realMatches)])
            Y.append(Y_fake[p[i] - len(realMatches)])

    # Write matched data and reordered data into a pickle file
    pickle_out = open('cleanData', "wb")
    pickle.dump(realMatches, pickle_out)
    pickle.dump(fakeMatches, pickle_out)
    pickle.dump(X, pickle_out)
    pickle.dump(Y, pickle_out)
    pickle_out.close()
