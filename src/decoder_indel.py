import pickle
from itertools import product
import numpy as np
from scipy import stats


# Function to decode the two-hot encoding of the cleaned data
# For viewing of sequence prediction results
def decode(sequence, dataset):

    if sequence not in range(len(dataset)):
        raise ValueError("Invalid index")
    else:
        tmp = dataset[sequence]
        ref = []
        cnd = []

        for j in range(len(tmp[0])):
            # if j < 10:
            #     print(tmp[0, j])
            if tmp[0, j, 0] == 1:
                ref.append('A')
            elif tmp[0, j, 1] == 1:
                ref.append('C')
            elif tmp[0, j, 2] == 1:
                ref.append('G')
            elif tmp[0, j, 3] == 1:
                ref.append('T')
            elif tmp[0, j, 4] == 1:
                ref.append('-')

            if tmp[0, j, 5] == 1:
                cnd.append('A')
            elif tmp[0, j, 6] == 1:
                cnd.append('C')
            elif tmp[0, j, 7] == 1:
                cnd.append('G')
            elif tmp[0, j, 8] == 1:
                cnd.append('T')
            elif tmp[0, j, 9] == 1:
                cnd.append('-')

            elif tmp[0, j, 10] == 1:
                cnd.append('iAAn')
            elif tmp[0, j, 11] == 1:
                cnd.append('iACn')
            elif tmp[0, j, 12] == 1:
                cnd.append('iAGn')
            elif tmp[0, j, 13] == 1:
                cnd.append('iATn')
            elif tmp[0, j, 14] == 1:
                cnd.append('iCAn')
            elif tmp[0, j, 15] == 1:
                cnd.append('iCCn')
            elif tmp[0, j, 16] == 1:
                cnd.append('iCGn')
            elif tmp[0, j, 17] == 1:
                cnd.append('iCTn')
            elif tmp[0, j, 18] == 1:
                cnd.append('iGAn')
            elif tmp[0, j, 19] == 1:
                cnd.append('iGCn')
            elif tmp[0, j, 20] == 1:
                cnd.append('iGGn')
            elif tmp[0, j, 21] == 1:
                cnd.append('iGTn')
            elif tmp[0, j, 22] == 1:
                cnd.append('iTAn')
            elif tmp[0, j, 23] == 1:
                cnd.append('iTCn')
            elif tmp[0, j, 24] == 1:
                cnd.append('iTGn')
            elif tmp[0, j, 25] == 1:
                cnd.append('iTTn')

            elif tmp[0, j, 26] == 1:
                cnd.append('iAAAn')
            elif tmp[0, j, 27] == 1:
                cnd.append('iAACn')
            elif tmp[0, j, 28] == 1:
                cnd.append('iAAGn')
            elif tmp[0, j, 29] == 1:
                cnd.append('iAATn')
            elif tmp[0, j, 30] == 1:
                cnd.append('iACAn')
            elif tmp[0, j, 31] == 1:
                cnd.append('iACCn')
            elif tmp[0, j, 32] == 1:
                cnd.append('iACGn')
            elif tmp[0, j, 33] == 1:
                cnd.append('iACTn')
            elif tmp[0, j, 34] == 1:
                cnd.append('iAGAn')
            elif tmp[0, j, 35] == 1:
                cnd.append('iAGCn')
            elif tmp[0, j, 36] == 1:
                cnd.append('iAGGn')
            elif tmp[0, j, 37] == 1:
                cnd.append('iAGTn')
            elif tmp[0, j, 38] == 1:
                cnd.append('iATAn')
            elif tmp[0, j, 39] == 1:
                cnd.append('iATCn')
            elif tmp[0, j, 40] == 1:
                cnd.append('iATGn')
            elif tmp[0, j, 41] == 1:
                cnd.append('iATTn')

            elif tmp[0, j, 42] == 1:
                cnd.append('iCAAn')
            elif tmp[0, j, 43] == 1:
                cnd.append('iCACn')
            elif tmp[0, j, 44] == 1:
                cnd.append('iCAGn')
            elif tmp[0, j, 45] == 1:
                cnd.append('iCATn')
            elif tmp[0, j, 46] == 1:
                cnd.append('iCCAn')
            elif tmp[0, j, 47] == 1:
                cnd.append('iCCCn')
            elif tmp[0, j, 48] == 1:
                cnd.append('iCCGn')
            elif tmp[0, j, 49] == 1:
                cnd.append('iCCTn')
            elif tmp[0, j, 50] == 1:
                cnd.append('iCGAn')
            elif tmp[0, j, 51] == 1:
                cnd.append('iCGCn')
            elif tmp[0, j, 52] == 1:
                cnd.append('iCGGn')
            elif tmp[0, j, 53] == 1:
                cnd.append('iCGTn')
            elif tmp[0, j, 54] == 1:
                cnd.append('iCTAn')
            elif tmp[0, j, 55] == 1:
                cnd.append('iCTCn')
            elif tmp[0, j, 56] == 1:
                cnd.append('iCTGn')
            elif tmp[0, j, 57] == 1:
                cnd.append('iCTTn')

            elif tmp[0, j, 58] == 1:
                cnd.append('iGAAn')
            elif tmp[0, j, 59] == 1:
                cnd.append('iGACn')
            elif tmp[0, j, 60] == 1:
                cnd.append('iGAGn')
            elif tmp[0, j, 61] == 1:
                cnd.append('iGATn')
            elif tmp[0, j, 62] == 1:
                cnd.append('iGCAn')
            elif tmp[0, j, 63] == 1:
                cnd.append('iGCCn')
            elif tmp[0, j, 64] == 1:
                cnd.append('iGCGn')
            elif tmp[0, j, 65] == 1:
                cnd.append('iGCTn')
            elif tmp[0, j, 66] == 1:
                cnd.append('iGGAn')
            elif tmp[0, j, 67] == 1:
                cnd.append('iGGCn')
            elif tmp[0, j, 68] == 1:
                cnd.append('iGGGn')
            elif tmp[0, j, 69] == 1:
                cnd.append('iGGTn')
            elif tmp[0, j, 70] == 1:
                cnd.append('iGTAn')
            elif tmp[0, j, 71] == 1:
                cnd.append('iGTCn')
            elif tmp[0, j, 72] == 1:
                cnd.append('iGTGn')
            elif tmp[0, j, 73] == 1:
                cnd.append('iGTTn')

            elif tmp[0, j, 74] == 1:
                cnd.append('iTAAn')
            elif tmp[0, j, 75] == 1:
                cnd.append('iTACn')
            elif tmp[0, j, 76] == 1:
                cnd.append('iTAGn')
            elif tmp[0, j, 77] == 1:
                cnd.append('iTATn')
            elif tmp[0, j, 78] == 1:
                cnd.append('iTCAn')
            elif tmp[0, j, 79] == 1:
                cnd.append('iTCCn')
            elif tmp[0, j, 80] == 1:
                cnd.append('iTCGn')
            elif tmp[0, j, 81] == 1:
                cnd.append('iTCTn')
            elif tmp[0, j, 82] == 1:
                cnd.append('iTGAn')
            elif tmp[0, j, 83] == 1:
                cnd.append('iTGCn')
            elif tmp[0, j, 84] == 1:
                cnd.append('iTGGn')
            elif tmp[0, j, 85] == 1:
                cnd.append('iTGTn')
            elif tmp[0, j, 86] == 1:
                cnd.append('iTTAn')
            elif tmp[0, j, 87] == 1:
                cnd.append('iTTCn')
            elif tmp[0, j, 88] == 1:
                cnd.append('iTTGn')
            elif tmp[0, j, 89] == 1:
                cnd.append('iTTTn')

            elif tmp[0, j, 90] == 1:
                cnd.append('iAXn')
            elif tmp[0, j, 91] == 1:
                cnd.append('iCXn')
            elif tmp[0, j, 92] == 1:
                cnd.append('iGXn')
            elif tmp[0, j, 93] == 1:
                cnd.append('iTXn')

            elif tmp[0, j, 94] == 1:
                cnd.append('Z')  # dummy

    return ref, cnd


# Reconstruct alignment given ancestor (gapless) and descendent sequence (with insertion chars)
def reconstruct(anc, des):

    anc_out = []
    des_out = []

    i = 0  # counter for ancestor
    j = 0  # counter for descendant
    while i < len(anc) and j < len(des):

        anc_out.append(anc[i])

        if des[j] != 'i':
            des_out.append(des[j])

        elif des[j] == 'i':
            k = 0
            temp = []
            while des[j + k] != 'n':
                temp.append(des[j + k])
                k += 1
            temp.append(des[j + k])
            j += k
            temp = ''.join(temp)  # now will reconstruct based on insertion character found

            if temp == 'iAAn':
                anc_out.append('-')
                des_out.append('A')
                des_out.append('A')
            elif temp == 'iACn':
                anc_out.append('-')
                des_out.append('A')
                des_out.append('C')
            elif temp == 'iAGn':
                anc_out.append('-')
                des_out.append('A')
                des_out.append('G')
            elif temp == 'iATn':
                anc_out.append('-')
                des_out.append('A')
                des_out.append('T')
            elif temp == 'iCAn':
                anc_out.append('-')
                des_out.append('C')
                des_out.append('A')
            elif temp == 'iCCn':
                anc_out.append('-')
                des_out.append('C')
                des_out.append('C')
            elif temp == 'iCGn':
                anc_out.append('-')
                des_out.append('C')
                des_out.append('G')
            elif temp == 'iCTn':
                anc_out.append('-')
                des_out.append('C')
                des_out.append('T')
            elif temp == 'iGAn':
                anc_out.append('-')
                des_out.append('G')
                des_out.append('A')
            elif temp == 'iGCn':
                anc_out.append('-')
                des_out.append('G')
                des_out.append('C')
            elif temp == 'iGGn':
                anc_out.append('-')
                des_out.append('G')
                des_out.append('G')
            elif temp == 'iGTn':
                anc_out.append('-')
                des_out.append('G')
                des_out.append('T')
            elif temp == 'iTAn':
                anc_out.append('-')
                des_out.append('T')
                des_out.append('A')
            elif temp == 'iTCn':
                anc_out.append('-')
                des_out.append('T')
                des_out.append('C')
            elif temp == 'iTGn':
                anc_out.append('-')
                des_out.append('T')
                des_out.append('G')
            elif temp == 'iTTn':
                anc_out.append('-')
                des_out.append('T')
                des_out.append('T')

            elif temp == 'iAAAn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('A')
                des_out.append('A')
                des_out.append('A')
            elif temp == 'iAACn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('A')
                des_out.append('A')
                des_out.append('C')
            elif temp == 'iAAGn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('A')
                des_out.append('A')
                des_out.append('G')
            elif temp == 'iAATn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('A')
                des_out.append('A')
                des_out.append('T')
            elif temp == 'iACAn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('A')
                des_out.append('C')
                des_out.append('A')
            elif temp == 'iACCn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('A')
                des_out.append('C')
                des_out.append('C')
            elif temp == 'iACGn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('A')
                des_out.append('C')
                des_out.append('G')
            elif temp == 'iACTn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('A')
                des_out.append('C')
                des_out.append('T')
            elif temp == 'iAGAn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('A')
                des_out.append('G')
                des_out.append('A')
            elif temp == 'iAGCn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('A')
                des_out.append('G')
                des_out.append('C')
            elif temp == 'iAGGn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('A')
                des_out.append('G')
                des_out.append('G')
            elif temp == 'iAGTn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('A')
                des_out.append('G')
                des_out.append('T')
            elif temp == 'iATAn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('A')
                des_out.append('T')
                des_out.append('A')
            elif temp == 'iATCn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('A')
                des_out.append('T')
                des_out.append('C')
            elif temp == 'iATGn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('A')
                des_out.append('T')
                des_out.append('G')
            elif temp == 'iATTn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('A')
                des_out.append('T')
                des_out.append('T')

            elif temp == 'iCAAn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('C')
                des_out.append('A')
                des_out.append('A')
            elif temp == 'iCACn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('C')
                des_out.append('A')
                des_out.append('C')
            elif temp == 'iCAGn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('C')
                des_out.append('A')
                des_out.append('G')
            elif temp == 'iCATn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('C')
                des_out.append('A')
                des_out.append('T')
            elif temp == 'iCCAn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('C')
                des_out.append('C')
                des_out.append('A')
            elif temp == 'iCCCn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('C')
                des_out.append('C')
                des_out.append('C')
            elif temp == 'iCCGn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('C')
                des_out.append('C')
                des_out.append('G')
            elif temp == 'iCCTn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('C')
                des_out.append('C')
                des_out.append('T')
            elif temp == 'iCGAn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('C')
                des_out.append('G')
                des_out.append('A')
            elif temp == 'iCGCn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('C')
                des_out.append('G')
                des_out.append('C')
            elif temp == 'iCGGn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('C')
                des_out.append('G')
                des_out.append('G')
            elif temp == 'iCGTn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('C')
                des_out.append('G')
                des_out.append('T')
            elif temp == 'iCTAn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('C')
                des_out.append('T')
                des_out.append('A')
            elif temp == 'iCTCn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('C')
                des_out.append('T')
                des_out.append('C')
            elif temp == 'iCTGn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('C')
                des_out.append('T')
                des_out.append('G')
            elif temp == 'iCTTn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('C')
                des_out.append('T')
                des_out.append('T')

            elif temp == 'iGAAn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('G')
                des_out.append('A')
                des_out.append('A')
            elif temp == 'iGACn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('G')
                des_out.append('A')
                des_out.append('C')
            elif temp == 'iGAGn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('G')
                des_out.append('A')
                des_out.append('G')
            elif temp == 'iGATn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('G')
                des_out.append('A')
                des_out.append('T')
            elif temp == 'iGCAn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('G')
                des_out.append('C')
                des_out.append('A')
            elif temp == 'iGCCn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('G')
                des_out.append('C')
                des_out.append('C')
            elif temp == 'iGCGn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('G')
                des_out.append('C')
                des_out.append('G')
            elif temp == 'iGCTn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('G')
                des_out.append('C')
                des_out.append('T')
            elif temp == 'iGGAn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('G')
                des_out.append('G')
                des_out.append('A')
            elif temp == 'iGGCn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('G')
                des_out.append('G')
                des_out.append('C')
            elif temp == 'iGGGn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('G')
                des_out.append('G')
                des_out.append('G')
            elif temp == 'iGGTn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('G')
                des_out.append('G')
                des_out.append('T')
            elif temp == 'iGTAn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('G')
                des_out.append('T')
                des_out.append('A')
            elif temp == 'iGTCn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('G')
                des_out.append('T')
                des_out.append('C')
            elif temp == 'iGTGn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('G')
                des_out.append('T')
                des_out.append('G')
            elif temp == 'iGTTn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('G')
                des_out.append('T')
                des_out.append('T')

            elif temp == 'iTAAn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('T')
                des_out.append('A')
                des_out.append('A')
            elif temp == 'iTACn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('T')
                des_out.append('A')
                des_out.append('C')
            elif temp == 'iTAGn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('T')
                des_out.append('A')
                des_out.append('G')
            elif temp == 'iTATn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('T')
                des_out.append('A')
                des_out.append('T')
            elif temp == 'iTCAn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('T')
                des_out.append('C')
                des_out.append('A')
            elif temp == 'iTCCn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('T')
                des_out.append('C')
                des_out.append('C')
            elif temp == 'iTCGn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('T')
                des_out.append('C')
                des_out.append('G')
            elif temp == 'iTCTn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('T')
                des_out.append('C')
                des_out.append('T')
            elif temp == 'iTGAn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('T')
                des_out.append('G')
                des_out.append('A')
            elif temp == 'iTGCn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('T')
                des_out.append('G')
                des_out.append('C')
            elif temp == 'iTGGn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('T')
                des_out.append('G')
                des_out.append('G')
            elif temp == 'iTGTn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('T')
                des_out.append('G')
                des_out.append('T')
            elif temp == 'iTTAn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('T')
                des_out.append('T')
                des_out.append('A')
            elif temp == 'iTTCn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('T')
                des_out.append('T')
                des_out.append('C')
            elif temp == 'iTTGn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('T')
                des_out.append('T')
                des_out.append('G')
            elif temp == 'iTTTn':
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('T')
                des_out.append('T')
                des_out.append('T')

            # Change to be probabilistic later based on real data
            # For now will assume X is insertion size |3|
            elif temp == 'iAXn':
                anc_out.append('-')
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('A')
                des_out.append('A')
                des_out.append('A')
                des_out.append('A')
            elif temp == 'iCXn':
                anc_out.append('-')
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('C')
                des_out.append('C')
                des_out.append('C')
                des_out.append('C')
            elif temp == 'iGXn':
                anc_out.append('-')
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('G')
                des_out.append('G')
                des_out.append('G')
                des_out.append('G')
            elif temp == 'iTXn':
                anc_out.append('-')
                anc_out.append('-')
                anc_out.append('-')
                des_out.append('T')
                des_out.append('T')
                des_out.append('T')
                des_out.append('T')

            else:  # do nothing with dummy character?
                continue

        i += 1
        j += 1

    anc_out = ''.join(anc_out)
    des_out = ''.join(des_out)

    return anc_out, des_out


def main(file_name):
    pickle_in = open(file_name, "rb")
    X_train = pickle.load(pickle_in)  # real matched sequence pairs
    print(len(X_train))
    pickle_in.close()

    posList = [i for i in range(0, len(X_train))]
    # posList = [i for i in range(0, 1000)]

    mutations_list = []
    insertions_list = []
    deletions_list = []
    conservation_types = {}
    mutation_types = {}
    for pos in posList:
        ref, cnd = decode(pos, X_train)

        fix_ref = ''
        fix_cnd = ''

        for i in ref:
            fix_ref += str(i)
        for i in cnd:
            fix_cnd += str(i)

        print(pos)
        print("Ancestral")
        print(fix_ref)
        print("Descendant")
        print(fix_cnd)
        print()

        anc, des = reconstruct(fix_ref, fix_cnd)

        print("After reconstruction...")
        print("Ancestral")
        print(anc)
        print("Descendant")
        print(des)
        print()

        num_insertions = 0
        num_deletions = 0
        num_mutations = 0
        for i in range(min(len(anc), len(des))):
            if anc[i] != des[i]:
                if des[i] == '-':
                    num_deletions += 1
                elif anc[i] == '-':
                    num_insertions += 1
                else:
                    num_mutations += 1

                temp = anc[i] + "->" + des[i]
                if temp in mutation_types:
                    mutation_types[temp] += 1
                else:
                    mutation_types[temp] = 1
            else:
                temp = anc[i] + "->" + des[i]
                if temp in conservation_types:
                    conservation_types[temp] += 1
                else:
                    conservation_types[temp] = 1

        mutations_list.append(num_mutations)
        insertions_list.append(num_insertions)
        deletions_list.append(num_deletions)

    print(mutations_list)
    print("Max number of mutations in alignment: ", np.max(mutations_list))
    print("Mean number of mutations in alignment: ", np.mean(mutations_list))
    print("Median number of mutations in alignment: ", np.median(mutations_list))
    print("Mode number of mutations in alignment: ", stats.mode(mutations_list)[0])

    print(insertions_list)
    print("Max number of insertions in alignment: ", np.max(insertions_list))
    print("Mean number of insertions in alignment: ", np.mean(insertions_list))
    print("Median number of insertions in alignment: ", np.median(insertions_list))
    print("Mode number of insertions in alignment: ", stats.mode(insertions_list)[0])

    print(deletions_list)
    print("Max number of deletions in alignment: ", np.max(deletions_list))
    print("Mean number of deletions in alignment: ", np.mean(deletions_list))
    print("Median number of deletions in alignment: ", np.median(deletions_list))
    print("Mode number of deletions in alignment: ", stats.mode(deletions_list)[0])

    print(conservation_types)
    print(mutation_types)

    total = 0
    total_A = 0
    total_C = 0
    total_G = 0
    total_T = 0
    total_gap = 0
    for c in conservation_types:
        total += conservation_types[c]
        if c.startswith('A'):
            total_A += conservation_types[c]
        elif c.startswith('C'):
            total_C += conservation_types[c]
        elif c.startswith('G'):
            total_G += conservation_types[c]
        elif c.startswith('T'):
            total_T += conservation_types[c]
        elif c.startswith('-'):
            total_gap += conservation_types[c]

    print(total)
    print(total_A)
    print(total_C)
    print(total_G)
    print(total_T)
    print(total_gap)
    for m in mutation_types:
        total += mutation_types[m]
        if m.startswith('A'):
            total_A += mutation_types[m]
        elif m.startswith('C'):
            total_C += mutation_types[m]
        elif m.startswith('G'):
            total_G += mutation_types[m]
        elif m.startswith('T'):
            total_T += mutation_types[m]
        elif m.startswith('-'):
            total_gap += mutation_types[m]
    print('Final total:', total)
    print(total_A)
    print(total_C)
    print(total_G)
    print(total_T)
    print(total_gap)

    mutation_probabilities = {}

    for c in conservation_types:
        mutation_probabilities[c] = conservation_types[c] / total
    for m in mutation_types:
        mutation_probabilities[m] = mutation_types[m] / total

    print(mutation_probabilities)

    for key in sorted(mutation_probabilities.keys()):
        print(key, " : ", mutation_probabilities[key])

    conditional_mutation_probabilities = {}

    for c in conservation_types:
        if c.startswith('A'):
            conditional_mutation_probabilities[c] = conservation_types[c] / total_A
        elif c.startswith('C'):
            conditional_mutation_probabilities[c] = conservation_types[c] / total_C
        elif c.startswith('G'):
            conditional_mutation_probabilities[c] = conservation_types[c] / total_G
        elif c.startswith('T'):
            conditional_mutation_probabilities[c] = conservation_types[c] / total_T
        elif c.startswith('-'):
            conditional_mutation_probabilities[c] = conservation_types[c] / total_gap
    for m in mutation_types:
        if m.startswith('A'):
            conditional_mutation_probabilities[m] = mutation_types[m] / total_A
        elif m.startswith('C'):
            conditional_mutation_probabilities[m] = mutation_types[m] / total_C
        elif m.startswith('G'):
            conditional_mutation_probabilities[m] = mutation_types[m] / total_G
        elif m.startswith('T'):
            conditional_mutation_probabilities[m] = mutation_types[m] / total_T
        elif m.startswith('-'):
            conditional_mutation_probabilities[m] = mutation_types[m] / total_gap

    print(conditional_mutation_probabilities)

    for key in sorted(conditional_mutation_probabilities.keys()):
        print(key, " : ", conditional_mutation_probabilities[key])


def count(file_name):
    pickle_in = open(file_name, "rb")
    X_train = pickle.load(pickle_in)  # real matched sequence pairs
    print(len(X_train))
    pickle_in.close()

    posList = [i for i in range(0, len(X_train))]
    # posList = [i for i in range(0, 2)]

    alphabet = ['A', 'C', 'G', 'T', '-']
    tri_mutation_types = {}
    di_mutation_types = {}
    single_mutation_types = {}
    tri = list(product(alphabet, repeat=3))  # trinucleotides
    di = list(product(alphabet, repeat=2))  # dinucleotides
    single = list(product(alphabet, repeat=1))  # single nucleotides
    for i in range(len(tri)):
        tri[i] = ''.join(tri[i])
    for i in range(len(di)):
        di[i] = ''.join(di[i])
    for i in range(len(single)):
        single[i] = ''.join(single[i])

    for i in range(len(tri)):
        for j in range(len(tri)):
            temp = tri[i] + '->' + tri[j]
            tri_mutation_types[temp] = 0
    print(len(tri_mutation_types))
    for i in range(len(di)):
        for j in range(len(di)):
            temp = di[i] + '->' + di[j]
            di_mutation_types[temp] = 0
    print(len(di_mutation_types))
    for i in range(len(single)):
        for j in range(len(single)):
            temp = single[i] + '->' + single[j]
            single_mutation_types[temp] = 0
    print(len(single_mutation_types))

    tri_count = 0
    di_count = 0
    single_count = 0
    for pos in posList:
        ref, cnd = decode(pos, X_train)

        fix_ref = ''
        fix_cnd = ''

        for i in ref:
            fix_ref += str(i)
        for i in cnd:
            fix_cnd += str(i)

        print(pos)
        print("Ancestral")
        print(fix_ref)
        print("Descendant")
        print(fix_cnd)
        print()

        anc, des = reconstruct(fix_ref, fix_cnd)

        print("After reconstruction...")
        print("Ancestral")
        print(anc)
        print("Descendant")
        print(des)
        print()

        for i in range(0, min(len(anc), len(des)) - 2, 1):
            temp_anc = anc[i] + anc[i + 1] + anc[i + 2]
            des_anc = des[i] + des[i + 1] + des[i + 2]
            temp = temp_anc + '->' + des_anc
            if temp in tri_mutation_types:
                tri_mutation_types[temp] += 1
            else:
                print("Unexpected mutation not found in dictionary")
                tri_mutation_types[temp] = 1
            tri_count += 1

        for i in range(0, min(len(anc), len(des)) - 1, 1):
            temp_anc = anc[i] + anc[i + 1]
            des_anc = des[i] + des[i + 1]
            temp = temp_anc + '->' + des_anc
            if temp in di_mutation_types:
                di_mutation_types[temp] += 1
            else:
                print("Unexpected mutation not found in dictionary")
                di_mutation_types[temp] = 1
            di_count += 1

        for i in range(0, min(len(anc), len(des)), 1):
            temp_anc = anc[i]
            des_anc = des[i]
            temp = temp_anc + '->' + des_anc
            if temp in single_mutation_types:
                single_mutation_types[temp] += 1
            else:
                print("Unexpected mutation not found in dictionary")
                single_mutation_types[temp] = 1
            single_count += 1

    print(tri_mutation_types)
    print(di_mutation_types)
    print(single_mutation_types)
    print()
    print(tri_count)
    print(di_count)
    print(single_count)
    print()

    # Increment by 1 for smoothing when later computing divergence values
    for tri in tri_mutation_types:
        tri_mutation_types[tri] = (tri_mutation_types[tri] + 1) / tri_count
    for di in di_mutation_types:
        di_mutation_types[di] = (di_mutation_types[di] + 1) / di_count
    for single in single_mutation_types:
        single_mutation_types[single] = (single_mutation_types[single] + 1) / single_count

    print(tri_mutation_types)
    print(di_mutation_types)
    print(single_mutation_types)

    pickle_out = open(file_name + '_extended_probabilities', "wb")
    pickle.dump(tri_mutation_types, pickle_out)
    pickle.dump(di_mutation_types, pickle_out)
    pickle.dump(single_mutation_types, pickle_out)
    pickle_out.close()


if __name__ == "__main__":

    # main('realData_indels_1500')
    count('realData_indels_1500')


# -----Some leftover code in case it's ever useful, check indices still work-----
# Separate ancestral from descendent in matched sequences
# Recall that ancestor is first 5 bits and descendent is next 5 bits for 2-hot encoding over 10 bits
# X_anc = []
# X_des = []
# for i in range(len(X)):
#     for j in range(len(X[i][0])):
#         temp = X[i][0][j]
#         # print(temp)
#         X_anc.append(temp[:5])
#         X_des.append(temp[5:])
