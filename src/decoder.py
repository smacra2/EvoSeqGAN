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
        ref = np.chararray(len(tmp[0]))
        cnd = np.chararray(len(tmp[0]))

        for j in range(len(tmp[0])):
            if tmp[0, j, 0] == 1:
                ref[j] = 'A'
            elif tmp[0, j, 1] == 1:
                ref[j] = 'C'
            elif tmp[0, j, 2] == 1:
                ref[j] = 'G'
            elif tmp[0, j, 3] == 1:
                ref[j] = 'T'
            else:
                ref[j] = '-'

            if tmp[0, j, 5] == 1:
                cnd[j] = 'A'
            elif tmp[0, j, 6] == 1:
                cnd[j] = 'C'
            elif tmp[0, j, 7] == 1:
                cnd[j] = 'G'
            elif tmp[0, j, 8] == 1:
                cnd[j] = 'T'
            else:
                cnd[j] = '-'

    return ref, cnd


def main(file_name):

    pickle_in = open(file_name, "rb")
    X_train = pickle.load(pickle_in)  # real matched sequence pairs
    print(len(X_train))
    pickle_in.close()

    # pickle_in = open('cleanData', "rb")
    # realMatches = pickle.load(pickle_in)  # real matched sequence pairs
    # fakeMatches = pickle.load(pickle_in)  # fake matched sequence pairs
    # X_train = pickle.load(pickle_in)  # encoded matched sequence pairs
    # Y_train = pickle.load(pickle_in)  # associated label: 1 for real, 0 for fake
    # pickle_in.close()

    # pickle_in = open('testData', "rb")
    # X_test = pickle.load(pickle_in)  # encoded matched sequence pairs
    # Y_test = pickle.load(pickle_in)  # associated label: 1 for real, 0 for fake
    # pickle_in.close()

    posList = [i for i in range(0, len(X_train))]

    mutations_list = []
    conservation_types = {}
    mutation_types = {}
    for pos in posList:
        ref, cnd = decode(pos, X_train)

        fix_ref = ''
        fix_cnd = ''

        for i in ref:
            fix_ref += str(i.decode('utf-8'))
        for i in cnd:
            fix_cnd += str(i.decode('utf-8'))

        print(pos)
        print("Ancestral")
        print(fix_ref)
        print("Descendant")
        print(fix_cnd)
        print()

        num_mutations = 0
        for i in range(min(len(fix_ref), len(fix_cnd))):
            if fix_ref[i] != fix_cnd[i]:
                num_mutations += 1
                temp = fix_ref[i] + "->" + fix_cnd[i]
                if temp in mutation_types:
                    mutation_types[temp] += 1
                else:
                    mutation_types[temp] = 1
            else:
                temp = fix_ref[i] + "->" + fix_cnd[i]
                if temp in conservation_types:
                    conservation_types[temp] += 1
                else:
                    conservation_types[temp] = 1

        mutations_list.append(num_mutations)

    print(mutations_list)
    print("Max number of mutations in alignment: ", np.max(mutations_list))
    print("Mean number of mutations in alignment: ", np.mean(mutations_list))
    print("Median number of mutations in alignment: ", np.median(mutations_list))
    print("Mode number of mutations in alignment: ", stats.mode(mutations_list)[0])

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

    alphabet = ['A', 'C', 'G', 'T']
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
            fix_ref += str(i.decode('utf-8'))
        for i in cnd:
            fix_cnd += str(i.decode('utf-8'))

        print(pos)
        print("Ancestral")
        print(fix_ref)
        print("Descendant")
        print(fix_cnd)
        print()

        anc, des = fix_ref, fix_cnd

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

    # main('realData_gapless_15000')
    count('realData_gapless_15000')


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
