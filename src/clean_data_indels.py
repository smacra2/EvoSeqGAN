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

    # print(temp_ref)
    # print(temp_cnd)
    temp_ref = "".join(temp_ref)
    temp_ref = temp_ref.replace("REPLACE", "")
    temp_cnd = "".join(temp_cnd)
    temp_cnd = temp_cnd.replace("REPLACE", "")
    print("Length of temp_ref and temp_cnd after removing double gaps: %d %d" % (len(temp_ref), len(temp_cnd)))
    temp_ref = list(temp_ref)
    temp_cnd = list(temp_cnd)

    # Remove gaps from ref, then at same position in cnd
    # Denote insertion with dinucleotide, trinucleotide, or X char (bigger than |2| insertion) accordingly
    i = 0
    while i < min(len(temp_ref), len(temp_cnd)):
        if temp_ref[i] == '-':
            temp_ref[i] = "REPLACE"

            # Arbitrarily delete the character pair if there is gap at first char in ref (cannot create meta-nucleotide)
            # Or if this deletion occurred on last iteration
            if i == 0 or temp_cnd[i - 1] == 'REPLACE':
                temp_cnd[i] = "REPLACE"

            # If next character in ref is not gap, then only dinucleotide case
            # Remove cnd at current position i in post-processing by marking with CHANGE
            elif temp_ref[i + 1] != '-':
                if temp_cnd[i - 1] == 'A':
                    if temp_cnd[i] == 'A':
                        temp_cnd[i - 1] = 'iAAn'
                    elif temp_cnd[i] == 'C':
                        temp_cnd[i - 1] = 'iACn'
                    elif temp_cnd[i] == 'G':
                        temp_cnd[i - 1] = 'iAGn'
                    elif temp_cnd[i] == 'T':
                        temp_cnd[i - 1] = 'iATn'

                elif temp_cnd[i - 1] == 'C':
                    if temp_cnd[i] == 'A':
                        temp_cnd[i - 1] = 'iCAn'
                    elif temp_cnd[i] == 'C':
                        temp_cnd[i - 1] = 'iCCn'
                    elif temp_cnd[i] == 'G':
                        temp_cnd[i - 1] = 'iCGn'
                    elif temp_cnd[i] == 'T':
                        temp_cnd[i - 1] = 'iCTn'

                elif temp_cnd[i - 1] == 'G':
                    if temp_cnd[i] == 'A':
                        temp_cnd[i - 1] = 'iGAn'
                    elif temp_cnd[i] == 'C':
                        temp_cnd[i - 1] = 'iGCn'
                    elif temp_cnd[i] == 'G':
                        temp_cnd[i - 1] = 'iGGn'
                    elif temp_cnd[i] == 'T':
                        temp_cnd[i - 1] = 'iGTn'

                elif temp_cnd[i - 1] == 'T':
                    if temp_cnd[i] == 'A':
                        temp_cnd[i - 1] = 'iTAn'
                    elif temp_cnd[i] == 'C':
                        temp_cnd[i - 1] = 'iTCn'
                    elif temp_cnd[i] == 'G':
                        temp_cnd[i - 1] = 'iTGn'
                    elif temp_cnd[i] == 'T':
                        temp_cnd[i - 1] = 'iTTn'

                temp_cnd[i] = 'CHANGE'

            # If next character in ref is gap but not one after that, then trinucleotide case
            # Remove cnd at current position i and next position i+1 in post-processing by marking with CHANGE
            elif temp_ref[i + 1] == '-' and temp_ref[i + 2] != '-':

                if temp_cnd[i - 1] == 'A':

                    if temp_cnd[i + 1] == 'A':
                        if temp_cnd[i] == 'A':
                            temp_cnd[i - 1] = 'iAAAn'
                        elif temp_cnd[i] == 'C':
                            temp_cnd[i - 1] = 'iACAn'
                        elif temp_cnd[i] == 'G':
                            temp_cnd[i - 1] = 'iAGAn'
                        elif temp_cnd[i] == 'T':
                            temp_cnd[i - 1] = 'iATAn'

                    elif temp_cnd[i + 1] == 'C':
                        if temp_cnd[i] == 'A':
                            temp_cnd[i - 1] = 'iAACn'
                        elif temp_cnd[i] == 'C':
                            temp_cnd[i - 1] = 'iACCn'
                        elif temp_cnd[i] == 'G':
                            temp_cnd[i - 1] = 'iAGCn'
                        elif temp_cnd[i] == 'T':
                            temp_cnd[i - 1] = 'iATCn'

                    elif temp_cnd[i + 1] == 'G':
                        if temp_cnd[i] == 'A':
                            temp_cnd[i - 1] = 'iAAGn'
                        elif temp_cnd[i] == 'C':
                            temp_cnd[i - 1] = 'iACGn'
                        elif temp_cnd[i] == 'G':
                            temp_cnd[i - 1] = 'iAGGn'
                        elif temp_cnd[i] == 'T':
                            temp_cnd[i - 1] = 'iATGn'

                    elif temp_cnd[i + 1] == 'T':
                        if temp_cnd[i] == 'A':
                            temp_cnd[i - 1] = 'iAATn'
                        elif temp_cnd[i] == 'C':
                            temp_cnd[i - 1] = 'iACTn'
                        elif temp_cnd[i] == 'G':
                            temp_cnd[i - 1] = 'iAGTn'
                        elif temp_cnd[i] == 'T':
                            temp_cnd[i - 1] = 'iATTn'

                elif temp_cnd[i - 1] == 'C':

                    if temp_cnd[i + 1] == 'A':
                        if temp_cnd[i] == 'A':
                            temp_cnd[i - 1] = 'iCAAn'
                        elif temp_cnd[i] == 'C':
                            temp_cnd[i - 1] = 'iCCAn'
                        elif temp_cnd[i] == 'G':
                            temp_cnd[i - 1] = 'iCGAn'
                        elif temp_cnd[i] == 'T':
                            temp_cnd[i - 1] = 'iCTAn'

                    elif temp_cnd[i + 1] == 'C':
                        if temp_cnd[i] == 'A':
                            temp_cnd[i - 1] = 'iCACn'
                        elif temp_cnd[i] == 'C':
                            temp_cnd[i - 1] = 'iCCCn'
                        elif temp_cnd[i] == 'G':
                            temp_cnd[i - 1] = 'iCGCn'
                        elif temp_cnd[i] == 'T':
                            temp_cnd[i - 1] = 'iCTCn'

                    elif temp_cnd[i + 1] == 'G':
                        if temp_cnd[i] == 'A':
                            temp_cnd[i - 1] = 'iCAGn'
                        elif temp_cnd[i] == 'C':
                            temp_cnd[i - 1] = 'iCCGn'
                        elif temp_cnd[i] == 'G':
                            temp_cnd[i - 1] = 'iCGGn'
                        elif temp_cnd[i] == 'T':
                            temp_cnd[i - 1] = 'iCTGn'

                    elif temp_cnd[i + 1] == 'T':
                        if temp_cnd[i] == 'A':
                            temp_cnd[i - 1] = 'iCATn'
                        elif temp_cnd[i] == 'C':
                            temp_cnd[i - 1] = 'iCCTn'
                        elif temp_cnd[i] == 'G':
                            temp_cnd[i - 1] = 'iCGTn'
                        elif temp_cnd[i] == 'T':
                            temp_cnd[i - 1] = 'iCTTn'

                elif temp_cnd[i - 1] == 'G':

                    if temp_cnd[i + 1] == 'A':
                        if temp_cnd[i] == 'A':
                            temp_cnd[i - 1] = 'iGAAn'
                        elif temp_cnd[i] == 'C':
                            temp_cnd[i - 1] = 'iGCAn'
                        elif temp_cnd[i] == 'G':
                            temp_cnd[i - 1] = 'iGGAn'
                        elif temp_cnd[i] == 'T':
                            temp_cnd[i - 1] = 'iGTAn'

                    elif temp_cnd[i + 1] == 'C':
                        if temp_cnd[i] == 'A':
                            temp_cnd[i - 1] = 'iGACn'
                        elif temp_cnd[i] == 'C':
                            temp_cnd[i - 1] = 'iGCCn'
                        elif temp_cnd[i] == 'G':
                            temp_cnd[i - 1] = 'iGGCn'
                        elif temp_cnd[i] == 'T':
                            temp_cnd[i - 1] = 'iGTCn'

                    elif temp_cnd[i + 1] == 'G':
                        if temp_cnd[i] == 'A':
                            temp_cnd[i - 1] = 'iGAGn'
                        elif temp_cnd[i] == 'C':
                            temp_cnd[i - 1] = 'iGCGn'
                        elif temp_cnd[i] == 'G':
                            temp_cnd[i - 1] = 'iGGGn'
                        elif temp_cnd[i] == 'T':
                            temp_cnd[i - 1] = 'iGTGn'

                    elif temp_cnd[i + 1] == 'T':
                        if temp_cnd[i] == 'A':
                            temp_cnd[i - 1] = 'iGATn'
                        elif temp_cnd[i] == 'C':
                            temp_cnd[i - 1] = 'iGCTn'
                        elif temp_cnd[i] == 'G':
                            temp_cnd[i - 1] = 'iGGTn'
                        elif temp_cnd[i] == 'T':
                            temp_cnd[i - 1] = 'iGTTn'

                elif temp_cnd[i - 1] == 'T':

                    if temp_cnd[i + 1] == 'A':
                        if temp_cnd[i] == 'A':
                            temp_cnd[i - 1] = 'iTAAn'
                        elif temp_cnd[i] == 'C':
                            temp_cnd[i - 1] = 'iTCAn'
                        elif temp_cnd[i] == 'G':
                            temp_cnd[i - 1] = 'iTGAn'
                        elif temp_cnd[i] == 'T':
                            temp_cnd[i - 1] = 'iTTAn'

                    elif temp_cnd[i + 1] == 'C':
                        if temp_cnd[i] == 'A':
                            temp_cnd[i - 1] = 'iTACn'
                        elif temp_cnd[i] == 'C':
                            temp_cnd[i - 1] = 'iTCCn'
                        elif temp_cnd[i] == 'G':
                            temp_cnd[i - 1] = 'iTGCn'
                        elif temp_cnd[i] == 'T':
                            temp_cnd[i - 1] = 'iTTCn'

                    elif temp_cnd[i + 1] == 'G':
                        if temp_cnd[i] == 'A':
                            temp_cnd[i - 1] = 'iTAGn'
                        elif temp_cnd[i] == 'C':
                            temp_cnd[i - 1] = 'iTCGn'
                        elif temp_cnd[i] == 'G':
                            temp_cnd[i - 1] = 'iTGGn'
                        elif temp_cnd[i] == 'T':
                            temp_cnd[i - 1] = 'iTTGn'

                    elif temp_cnd[i + 1] == 'T':
                        if temp_cnd[i] == 'A':
                            temp_cnd[i - 1] = 'iTATn'
                        elif temp_cnd[i] == 'C':
                            temp_cnd[i - 1] = 'iTCTn'
                        elif temp_cnd[i] == 'G':
                            temp_cnd[i - 1] = 'iTGTn'
                        elif temp_cnd[i] == 'T':
                            temp_cnd[i - 1] = 'iTTTn'

                temp_cnd[i] = 'CHANGE'
                temp_cnd[i + 1] = 'CHANGE'
                temp_ref[i + 1] = 'REPLACE'
                i += 1

            # Else use X char to denote bigger >= |3| insertion, X will later take on number and {A, C, G, T} char
            # According to distribution in real data
            elif temp_ref[i + 1] == '-' and temp_ref[i + 2] == '-':
                if temp_cnd[i - 1] == 'A':
                    temp_cnd[i - 1] = 'iAXn'
                elif temp_cnd[i - 1] == 'C':
                    temp_cnd[i - 1] = 'iCXn'
                elif temp_cnd[i - 1] == 'G':
                    temp_cnd[i - 1] = 'iGXn'
                elif temp_cnd[i - 1] == 'T':
                    temp_cnd[i - 1] = 'iTXn'

                # Skip over all subsequent gaps (generally expect j > 2 to be very rare)
                temp_cnd[i] = 'CHANGE'
                j = 1
                while temp_ref[i + j] == '-':
                    temp_ref[i + j] = 'REPLACE'
                    temp_cnd[i + j] = 'CHANGE'
                    j += 1
                i = i + j - 1  # Set counter i to last known position - 1 since will be +=1 on next line by default

        i += 1

    temp_ref = "".join(temp_ref)
    temp_ref = temp_ref.replace("REPLACE", "")
    temp_cnd = "".join(temp_cnd)
    temp_cnd = temp_cnd.replace("REPLACE", "")
    temp_cnd = temp_cnd.replace("CHANGE", "")
    print("Length of temp_ref and temp_cnd after removing gaps from ref and creating insertion chars: %d %d"
          % (len(temp_ref), len(temp_cnd)))

    # print(temp_ref)
    # print(temp_cnd)

    split = cut_length  # cut every 100 pieces to account for variable length of initial data
    # Ref is easy to cut, since only {A, C, G, T}
    match_ref = [temp_ref[i:i+split] for i in range(0, len(temp_ref), split)]

    # Cnd requires more work to treat dinucleotide, trinucleotide, and X chars as 'single' char
    # Can be 4 (di and X, ex. iAXn or iAAn) or 5 (tri, ex. iAAAn) reduced to 'single' char
    # match_cnd = [temp_cnd[i:i+split] for i in range(0, len(temp_cnd), split)]
    match_cnd = []
    chunk = []
    i = 0
    count = 0
    while i < len(temp_cnd):

        if temp_cnd[i] != 'i':
            chunk.append(temp_cnd[i])
            count += 1
        elif temp_cnd[i] == 'i':  # in special case
            j = 0
            while temp_cnd[i + j] != 'n':
                chunk.append(temp_cnd[i + j])
                j += 1
            chunk.append(temp_cnd[i + j])
            count += 1  # treat as single char
            i += j

        if count == split:
            temp = "".join(chunk)
            match_cnd.append(temp)
            chunk = []  # reset for next chunk
            count = 0

        i += 1

    # print(match_ref)
    # print(match_cnd)

    # Before appending to output list, remove alignments that are composed of more than 50% gaps
    matches = []
    print('Length match_ref: ', len(match_ref))
    print('Length match_cnd: ', len(match_cnd))
    removed_ref_count = 0
    removed_cnd_count = 0
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

        if num_gap_ref > (cut_length / 2) >= num_gap_cnd:
            removed_ref_count += 1
        elif num_gap_cnd > (cut_length / 2):
            removed_cnd_count += 1
        else:
            matches.append([match_ref[i], match_cnd[i]])

    print("Number of alignments removed for having >50%% gaps (for ref, for cnd): %d %d"
          % (removed_ref_count, removed_cnd_count))

    # Return the list of matched sequences
    return matches


# Function to one-hot encode the sequences in matches list and label with yhat
# A: 10000, C: 01000, G: 00100, T: 00010, -: 00001
# 16 dinucleotides, 64 trinucleotides, X big char
def prepare_data(matches, yhat):
    X = []
    Y = []

    # For each matched sequence in matches list
    for i in range(0, len(matches)):
        ref = matches[i][0]
        cnd = matches[i][1]
        # print(ref)
        # print(cnd)
        # Initialize encoding sequence
        tmp = np.zeros((1, len(ref), 95))  # 5 in ref + 90 in cnd
        # Encode reference sequence characters
        j = 0  # counter for ref and tmp
        k = 0  # counter for cnd to consider insertion chunks as single char
        while j < len(ref) and k < len(cnd):
            # print(ref[j])
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

            if cnd[k] != 'i':
                # print(cnd[k])
                if cnd[k] == 'A':
                    tmp[0, j, 5] = 1
                elif cnd[k] == 'C':
                    tmp[0, j, 6] = 1
                elif cnd[k] == 'G':
                    tmp[0, j, 7] = 1
                elif cnd[k] == 'T':
                    tmp[0, j, 8] = 1
                elif cnd[k] == '-':
                    tmp[0, j, 9] = 1

            elif cnd[k] == 'i':
                m = 0
                temp = []
                while cnd[k + m] != 'n':
                    # print('IN LOOP ', cnd[k+m])
                    temp.append(cnd[k + m])
                    m += 1
                temp.append(cnd[k + m])
                # print('IN LOOP ', cnd[k + m])
                k += m
                temp = ''.join(temp)
                # print(temp + '\n')

                if temp == 'iAAn':
                    tmp[0, j, 10] = 1
                elif temp == 'iACn':
                    tmp[0, j, 11] = 1
                elif temp == 'iAGn':
                    tmp[0, j, 12] = 1
                elif temp == 'iATn':
                    tmp[0, j, 13] = 1
                elif temp == 'iCAn':
                    tmp[0, j, 14] = 1
                elif temp == 'iCCn':
                    tmp[0, j, 15] = 1
                elif temp == 'iCGn':
                    tmp[0, j, 16] = 1
                elif temp == 'iCTn':
                    tmp[0, j, 17] = 1
                elif temp == 'iGAn':
                    tmp[0, j, 18] = 1
                elif temp == 'iGCn':
                    tmp[0, j, 19] = 1
                elif temp == 'iGGn':
                    tmp[0, j, 20] = 1
                elif temp == 'iGTn':
                    tmp[0, j, 21] = 1
                elif temp == 'iTAn':
                    tmp[0, j, 22] = 1
                elif temp == 'iTCn':
                    tmp[0, j, 23] = 1
                elif temp == 'iTGn':
                    tmp[0, j, 24] = 1
                elif temp == 'iTTn':
                    tmp[0, j, 25] = 1

                elif temp == 'iAAAn':
                    tmp[0, j, 26] = 1
                elif temp == 'iAACn':
                    tmp[0, j, 27] = 1
                elif temp == 'iAAGn':
                    tmp[0, j, 28] = 1
                elif temp == 'iAATn':
                    tmp[0, j, 29] = 1
                elif temp == 'iACAn':
                    tmp[0, j, 30] = 1
                elif temp == 'iACCn':
                    tmp[0, j, 31] = 1
                elif temp == 'iACGn':
                    tmp[0, j, 32] = 1
                elif temp == 'iACTn':
                    tmp[0, j, 33] = 1
                elif temp == 'iAGAn':
                    tmp[0, j, 34] = 1
                elif temp == 'iAGCn':
                    tmp[0, j, 35] = 1
                elif temp == 'iAGGn':
                    tmp[0, j, 36] = 1
                elif temp == 'iAGTn':
                    tmp[0, j, 37] = 1
                elif temp == 'iATAn':
                    tmp[0, j, 38] = 1
                elif temp == 'iATCn':
                    tmp[0, j, 39] = 1
                elif temp == 'iATGn':
                    tmp[0, j, 40] = 1
                elif temp == 'iATTn':
                    tmp[0, j, 41] = 1

                elif temp == 'iCAAn':
                    tmp[0, j, 42] = 1
                elif temp == 'iCACn':
                    tmp[0, j, 43] = 1
                elif temp == 'iCAGn':
                    tmp[0, j, 44] = 1
                elif temp == 'iCATn':
                    tmp[0, j, 45] = 1
                elif temp == 'iCCAn':
                    tmp[0, j, 46] = 1
                elif temp == 'iCCCn':
                    tmp[0, j, 47] = 1
                elif temp == 'iCCGn':
                    tmp[0, j, 48] = 1
                elif temp == 'iCCTn':
                    tmp[0, j, 49] = 1
                elif temp == 'iCGAn':
                    tmp[0, j, 50] = 1
                elif temp == 'iCGCn':
                    tmp[0, j, 51] = 1
                elif temp == 'iCGGn':
                    tmp[0, j, 52] = 1
                elif temp == 'iCGTn':
                    tmp[0, j, 53] = 1
                elif temp == 'iCTAn':
                    tmp[0, j, 54] = 1
                elif temp == 'iCTCn':
                    tmp[0, j, 55] = 1
                elif temp == 'iCTGn':
                    tmp[0, j, 56] = 1
                elif temp == 'iCTTn':
                    tmp[0, j, 57] = 1

                elif temp == 'iGAAn':
                    tmp[0, j, 58] = 1
                elif temp == 'iGACn':
                    tmp[0, j, 59] = 1
                elif temp == 'iGAGn':
                    tmp[0, j, 60] = 1
                elif temp == 'iGATn':
                    tmp[0, j, 61] = 1
                elif temp == 'iGCAn':
                    tmp[0, j, 62] = 1
                elif temp == 'iGCCn':
                    tmp[0, j, 63] = 1
                elif temp == 'iGCGn':
                    tmp[0, j, 64] = 1
                elif temp == 'iGCTn':
                    tmp[0, j, 65] = 1
                elif temp == 'iGGAn':
                    tmp[0, j, 66] = 1
                elif temp == 'iGGCn':
                    tmp[0, j, 67] = 1
                elif temp == 'iGGGn':
                    tmp[0, j, 68] = 1
                elif temp == 'iGGTn':
                    tmp[0, j, 69] = 1
                elif temp == 'iGTAn':
                    tmp[0, j, 70] = 1
                elif temp == 'iGTCn':
                    tmp[0, j, 71] = 1
                elif temp == 'iGTGn':
                    tmp[0, j, 72] = 1
                elif temp == 'iGTTn':
                    tmp[0, j, 73] = 1

                elif temp == 'iTAAn':
                    tmp[0, j, 74] = 1
                elif temp == 'iTACn':
                    tmp[0, j, 75] = 1
                elif temp == 'iTAGn':
                    tmp[0, j, 76] = 1
                elif temp == 'iTATn':
                    tmp[0, j, 77] = 1
                elif temp == 'iTCAn':
                    tmp[0, j, 78] = 1
                elif temp == 'iTCCn':
                    tmp[0, j, 79] = 1
                elif temp == 'iTCGn':
                    tmp[0, j, 80] = 1
                elif temp == 'iTCTn':
                    tmp[0, j, 81] = 1
                elif temp == 'iTGAn':
                    tmp[0, j, 82] = 1
                elif temp == 'iTGCn':
                    tmp[0, j, 83] = 1
                elif temp == 'iTGGn':
                    tmp[0, j, 84] = 1
                elif temp == 'iTGTn':
                    tmp[0, j, 85] = 1
                elif temp == 'iTTAn':
                    tmp[0, j, 86] = 1
                elif temp == 'iTTCn':
                    tmp[0, j, 87] = 1
                elif temp == 'iTTGn':
                    tmp[0, j, 88] = 1
                elif temp == 'iTTTn':
                    tmp[0, j, 89] = 1

                elif temp == 'iAXn':
                    tmp[0, j, 90] = 1
                elif temp == 'iCXn':
                    tmp[0, j, 91] = 1
                elif temp == 'iGXn':
                    tmp[0, j, 92] = 1
                elif temp == 'iTXn':
                    tmp[0, j, 93] = 1

                else:
                    tmp[0, j, 94] = 1

            j += 1
            k += 1

        # Add encoded sequences together with provided label
        X.append(tmp)
        Y.append([yhat])

    return X, Y


# Function to prepare only chunks of X length of real data
def real_data(chunk_size=500):
    realMatches = read_matches('Real_Alignments_48.000.000_Lines.txt', 's hg38', 's _HPGPNRMPC ', chunk_size)
    print("Number of real paired sequences: " + str(len(realMatches)))

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

    pickle_out = open('realData_indels_' + str(chunk_size), "wb")
    pickle.dump(X_train, pickle_out)
    pickle.dump(Y_train, pickle_out)
    pickle_out.close()

    for i in range(size_train_set, len(realMatches)):
        X_test.append(X_real[i])
        Y_test.append(Y_real[i])

    print("Size X_test: " + str(len(X_test)))
    print("Size Y_test: " + str(len(Y_test)))

    pickle_out = open('testData_indels_' + str(chunk_size), "wb")
    pickle.dump(X_test, pickle_out)
    pickle.dump(Y_test, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    real_data(1500)
