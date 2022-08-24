import math
import pickle
from collections import OrderedDict
from itertools import product
import numpy as np
import matplotlib.pyplot as plt


def count_mutations(file_name, flag='des', attention_val=5):
    f = open(file_name)
    data = []
    for row in f:
        if row[0] != '\n' and row != '--------------------\n':
            data.append(row.rstrip())
    print(len(data))
    # for i in range(len(data)):
    #     print(data[i])

    if flag == 'des':
        mutations_tracker = []
        mutation_types = {}
        conservation_types = {}
        for i in range(0, len(data), 2):
            anc = data[i]
            des = data[i + 1]

            num_mutations = 0
            for k in range(len(anc)):
                if anc[k] != des[k]:
                    num_mutations += 1
                    temp = anc[k] + "->" + des[k]
                    if temp in mutation_types:
                        mutation_types[temp] += 1
                    else:
                        mutation_types[temp] = 1
                else:
                    temp = anc[k] + "->" + des[k]
                    if temp in conservation_types:
                        conservation_types[temp] += 1
                    else:
                        conservation_types[temp] = 1

            # if num_mutations > attention_val:
            #     print("ATTENTION")
            # if num_mutations > 0:
            #     print(anc + '\n' + des)

            mutations_tracker.append(num_mutations)

        print(mutations_tracker)
        print("Number of sequences compared: ", len(mutations_tracker))
        print("Mean number of mutations over all sequences: ", np.mean(mutations_tracker))
        print("Median number of mutations over all sequences: ", np.median(mutations_tracker))
        print("Max number of mutations in 1 sequence: ", np.amax(mutations_tracker))

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

        # print(mutation_probabilities)

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

        # print(conditional_mutation_probabilities)

        return mutation_probabilities, conditional_mutation_probabilities

    elif flag == 'anc':
        mutations_tracker = []
        mutation_types = {}
        conservation_types = {}
        for i in range(0, len(data), 3):
            ori_anc = data[i]
            anc = data[i + 1]
            des = data[i + 2]

            num_mutations = 0
            for k in range(len(anc)):
                if anc[k] != des[k]:
                    num_mutations += 1
                    temp = anc[k] + "->" + des[k]
                    if temp in mutation_types:
                        mutation_types[temp] += 1
                    else:
                        mutation_types[temp] = 1
                else:
                    temp = anc[k] + "->" + des[k]
                    if temp in conservation_types:
                        conservation_types[temp] += 1
                    else:
                        conservation_types[temp] = 1

            if num_mutations > attention_val:
                print("ATTENTION")
            if num_mutations > 0:
                print(anc + '\n' + des)

            mutations_tracker.append(num_mutations)

        print(mutations_tracker)
        print("Number of sequences compared: ", len(mutations_tracker))
        print("Mean number of mutations over all sequences: ", np.mean(mutations_tracker))
        print("Median number of mutations over all sequences: ", np.median(mutations_tracker))
        print("Max number of mutations in 1 sequence: ", np.amax(mutations_tracker))

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

        # print(mutation_probabilities)

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

        # print(conditional_mutation_probabilities)

        return mutation_probabilities, conditional_mutation_probabilities


def main():
    probabilities, conditional_probabilities = count_mutations('generator_output_test_des.txt', 'des', 409)
    # count_mutations('generator_output_test_align.txt', 'anc', 10)

    reference = {'G->G': 0.20807706855791963, 'T->T': 0.2799418439716312, 'C->C': 0.20694278959810875,
                 'A->A': 0.2777275413711584, 'G->A': 0.004032033096926714, 'A->G': 0.0055735224586288415,
                 'A->T': 0.0008677304964539007, 'C->G': 0.0010501182033096927, 'T->A': 0.0008914893617021276,
                 'T->C': 0.00559290780141844, 'G->T': 0.0010936170212765958, 'T->G': 0.0010725768321513003,
                 'A->C': 0.0010544917257683216, 'C->T': 0.003909101654846336, 'C->A': 0.0011046099290780143,
                 'G->C': 0.0010685579196217493}

    condition_reference = {'G->G': 0.9710917481043837, 'T->T': 0.973714764990297, 'C->C': 0.9715322002446122,
                           'A->A': 0.9737197310058189, 'G->A': 0.018817422292833758, 'A->G': 0.019540909635311226,
                           'A->T': 0.003042284900594242, 'C->G': 0.004929979201303414, 'T->A': 0.003100845311317962,
                           'T->C': 0.019453672334966945, 'G->T': 0.0051038899784028, 'T->G': 0.0037307173634180835,
                           'A->C': 0.0036970744582756074, 'C->T': 0.018352019604491808, 'C->A': 0.005185800949592572,
                           'G->C': 0.0049869396243797355}

    # Ordered for convenience
    # A->A: 0.2777275413711584
    # A->C: 0.0010544917257683216
    # A->G: 0.0055735224586288415
    # A->T: 0.0008677304964539007
    # C->A: 0.0011046099290780143
    # C->C: 0.20694278959810875
    # C->G: 0.0010501182033096927
    # C->T: 0.003909101654846336
    # G->A: 0.004032033096926714
    # G->C: 0.0010685579196217493
    # G->G: 0.20807706855791963
    # G->T: 0.0010936170212765958
    # T->A: 0.0008914893617021276
    # T->C: 0.00559290780141844
    # T->G: 0.0010725768321513003
    # T->T: 0.2799418439716312
    #
    # Conditional probability by ancestor char
    # A->A: 0.9737197310058189
    # A->C: 0.0036970744582756074
    # A->G: 0.019540909635311226
    # A->T: 0.003042284900594242
    # C->A: 0.005185800949592572
    # C->C: 0.9715322002446122
    # C->G: 0.004929979201303414
    # C->T: 0.018352019604491808
    # G->A: 0.018817422292833758
    # G->C: 0.0049869396243797355
    # G->G: 0.9710917481043837
    # G->T: 0.0051038899784028
    # T->A: 0.003100845311317962
    # T->C: 0.019453672334966945
    # T->G: 0.0037307173634180835
    # T->T: 0.973714764990297

    for key in sorted(probabilities.keys()):
        print(key, " : ", probabilities[key])

    # Make sure every substitution in reference dictionary is in generated dictionary, even with probability 0
    for key in reference:
        if key not in probabilities:
            probabilities[key] = 0
            print("Missing key: ", key)

    euclidean_distance = 0
    for key in probabilities:
        for key2 in reference:
            if key == key2:
                euclidean_distance += (reference[key] - probabilities[key]) ** 2
    print("Squared Euclidean Distance: ", euclidean_distance)
    euclidean_distance = math.sqrt(euclidean_distance)
    print("Euclidean Distance: ", euclidean_distance)

    kl_divergence = 0
    for key in probabilities:
        for key2 in reference:
            if key == key2:
                temp = math.log((probabilities[key] / reference[key]) + 1e-8) * probabilities[key]
                kl_divergence += temp
    print("K-L Divergence: ", kl_divergence)

    print("\nNow repeating for conditional probabilities....")

    for key in sorted(conditional_probabilities.keys()):
        print(key, " : ", conditional_probabilities[key])

    for key in condition_reference:
        if key not in conditional_probabilities:
            conditional_probabilities[key] = 0
            print("Missing key: ", key)

    euclidean_distance = 0
    for key in conditional_probabilities:
        for key2 in condition_reference:
            if key == key2:
                euclidean_distance += (condition_reference[key] - conditional_probabilities[key]) ** 2
    print("Squared Euclidean Distance: ", euclidean_distance)
    euclidean_distance = math.sqrt(euclidean_distance)
    print("Euclidean Distance: ", euclidean_distance)

    kl_divergence = 0
    for key in conditional_probabilities:
        for key2 in condition_reference:
            if key == key2:
                temp = math.log((conditional_probabilities[key] / condition_reference[key]) + 1e-8) * \
                       conditional_probabilities[key]
                kl_divergence += temp
    print("K-L Divergence: ", kl_divergence)


def count_extended_mutations(file_name):
    f = open(file_name)
    data = []
    for row in f:
        if row[0] != '\n' and row != '--------------------\n':
            data.append(row.rstrip())
    print(len(data))

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
    for j in range(0, len(data), 2):
        anc = data[j]
        des = data[j + 1]

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

    return tri_mutation_types, di_mutation_types, single_mutation_types


def extended_main():
    pickle_in = open('realData_gapless_15000_extended_probabilities', "rb")
    # Load saved mutation probabilities from reference data
    tri_ref = pickle.load(pickle_in)
    di_ref = pickle.load(pickle_in)
    single_ref = pickle.load(pickle_in)
    pickle_in.close()

    tri_cnd, di_cnd, single_cnd = count_extended_mutations('generator_output_test_des.txt')

    # Make sure every substitution in reference dictionary is in generated dictionary, even with probability 0
    for key in tri_ref:
        if key not in tri_cnd:
            tri_cnd[key] = 0
            print("Missing key: ", key)
    for key in di_ref:
        if key not in di_cnd:
            di_cnd[key] = 0
            print("Missing key: ", key)
    for key in single_ref:
        if key not in single_cnd:
            single_cnd[key] = 0
            print("Missing key: ", key)

    print("\nFor trinucleotides...")

    euclidean_distance = 0
    for key in tri_cnd:
        for key2 in tri_ref:
            if key == key2:
                euclidean_distance += (tri_ref[key] - tri_cnd[key]) ** 2
    print("Squared Euclidean Distance: ", euclidean_distance)
    euclidean_distance = math.sqrt(euclidean_distance)
    print("Euclidean Distance: ", euclidean_distance)

    kl_divergence = 0
    for key in tri_cnd:
        for key2 in tri_ref:
            if key == key2:
                temp = math.log(((tri_cnd[key]) / (tri_ref[key]))) * tri_cnd[key]
                kl_divergence += temp
    print("K-L Divergence: ", kl_divergence)

    print("\nFor dinucleotides...")

    euclidean_distance = 0
    for key in di_cnd:
        for key2 in di_ref:
            if key == key2:
                euclidean_distance += (di_ref[key] - di_cnd[key]) ** 2
    print("Squared Euclidean Distance: ", euclidean_distance)
    euclidean_distance = math.sqrt(euclidean_distance)
    print("Euclidean Distance: ", euclidean_distance)

    kl_divergence = 0
    for key in di_cnd:
        for key2 in di_ref:
            if key == key2:
                temp = math.log(((di_cnd[key]) / (di_ref[key]))) * di_cnd[key]
                kl_divergence += temp
    print("K-L Divergence: ", kl_divergence)

    print("\nFor single nucleotides...")

    euclidean_distance = 0
    for key in single_cnd:
        for key2 in single_ref:
            if key == key2:
                euclidean_distance += (single_ref[key] - single_cnd[key]) ** 2
    print("Squared Euclidean Distance: ", euclidean_distance)
    euclidean_distance = math.sqrt(euclidean_distance)
    print("Euclidean Distance: ", euclidean_distance)

    kl_divergence = 0
    for key in single_cnd:
        for key2 in single_ref:
            if key == key2:
                temp = math.log(((single_cnd[key]) / (single_ref[key]))) * single_cnd[key]
                kl_divergence += temp
    print("K-L Divergence: ", kl_divergence)


def correlation_test(file_name):
    pickle_in = open('realData_gapless_15000_extended_probabilities', "rb")
    # Load saved mutation probabilities from reference data
    tri_ref = pickle.load(pickle_in)
    di_ref = pickle.load(pickle_in)
    single_ref = pickle.load(pickle_in)
    pickle_in.close()

    tri_cnd, di_cnd, single_cnd = count_extended_mutations(file_name)

    correlation_dict = {}  # dict of dict
    for i in range(16):
        correlation_dict[i] = OrderedDict()

    for ref_key in tri_ref.keys():

        # xyA -> xyA
        if ref_key[2] == 'A' and ref_key[7] == 'A':
            if ref_key in correlation_dict[0]:
                correlation_dict[0][ref_key] += tri_ref[ref_key]
            else:
                correlation_dict[0][ref_key] = tri_ref[ref_key]

        # xyA -> xyC
        elif ref_key[2] == 'A' and ref_key[7] == 'C':
            if ref_key in correlation_dict[1]:
                correlation_dict[1][ref_key] += tri_ref[ref_key]
            else:
                correlation_dict[1][ref_key] = tri_ref[ref_key]

        # xyA -> xyG
        elif ref_key[2] == 'A' and ref_key[7] == 'G':
            if ref_key in correlation_dict[2]:
                correlation_dict[2][ref_key] += tri_ref[ref_key]
            else:
                correlation_dict[2][ref_key] = tri_ref[ref_key]

        # xyA -> xyT
        elif ref_key[2] == 'A' and ref_key[7] == 'T':
            if ref_key in correlation_dict[3]:
                correlation_dict[3][ref_key] += tri_ref[ref_key]
            else:
                correlation_dict[3][ref_key] = tri_ref[ref_key]

        # xyC -> xyA
        elif ref_key[2] == 'C' and ref_key[7] == 'A':
            if ref_key in correlation_dict[4]:
                correlation_dict[4][ref_key] += tri_ref[ref_key]
            else:
                correlation_dict[4][ref_key] = tri_ref[ref_key]

        # xyC -> xyC
        elif ref_key[2] == 'C' and ref_key[7] == 'C':
            if ref_key in correlation_dict[5]:
                correlation_dict[5][ref_key] += tri_ref[ref_key]
            else:
                correlation_dict[5][ref_key] = tri_ref[ref_key]

        # xyC -> xyG
        elif ref_key[2] == 'C' and ref_key[7] == 'G':
            if ref_key in correlation_dict[6]:
                correlation_dict[6][ref_key] += tri_ref[ref_key]
            else:
                correlation_dict[6][ref_key] = tri_ref[ref_key]

        # xyC -> xyT
        elif ref_key[2] == 'C' and ref_key[7] == 'T':
            if ref_key in correlation_dict[7]:
                correlation_dict[7][ref_key] += tri_ref[ref_key]
            else:
                correlation_dict[7][ref_key] = tri_ref[ref_key]

        # xyG -> xyA
        elif ref_key[2] == 'G' and ref_key[7] == 'A':
            if ref_key in correlation_dict[8]:
                correlation_dict[8][ref_key] += tri_ref[ref_key]
            else:
                correlation_dict[8][ref_key] = tri_ref[ref_key]

        # xyG -> xyC
        elif ref_key[2] == 'G' and ref_key[7] == 'C':
            if ref_key in correlation_dict[9]:
                correlation_dict[9][ref_key] += tri_ref[ref_key]
            else:
                correlation_dict[9][ref_key] = tri_ref[ref_key]

        # xyG -> xyG
        elif ref_key[2] == 'G' and ref_key[7] == 'G':
            if ref_key in correlation_dict[10]:
                correlation_dict[10][ref_key] += tri_ref[ref_key]
            else:
                correlation_dict[10][ref_key] = tri_ref[ref_key]

        # xyG -> xyT
        elif ref_key[2] == 'G' and ref_key[7] == 'T':
            if ref_key in correlation_dict[11]:
                correlation_dict[11][ref_key] += tri_ref[ref_key]
            else:
                correlation_dict[11][ref_key] = tri_ref[ref_key]

        # xyT -> xyA
        elif ref_key[2] == 'T' and ref_key[7] == 'A':
            if ref_key in correlation_dict[12]:
                correlation_dict[12][ref_key] += tri_ref[ref_key]
            else:
                correlation_dict[12][ref_key] = tri_ref[ref_key]

        # xyT -> xyC
        elif ref_key[2] == 'T' and ref_key[7] == 'C':
            if ref_key in correlation_dict[13]:
                correlation_dict[13][ref_key] += tri_ref[ref_key]
            else:
                correlation_dict[13][ref_key] = tri_ref[ref_key]

        # xyT -> xyG
        elif ref_key[2] == 'T' and ref_key[7] == 'G':
            if ref_key in correlation_dict[14]:
                correlation_dict[14][ref_key] += tri_ref[ref_key]
            else:
                correlation_dict[14][ref_key] = tri_ref[ref_key]

        # xyT -> xyT
        elif ref_key[2] == 'T' and ref_key[7] == 'T':
            if ref_key in correlation_dict[15]:
                correlation_dict[15][ref_key] += tri_ref[ref_key]
            else:
                correlation_dict[15][ref_key] = tri_ref[ref_key]

    print()
    print(correlation_dict)

    ref_correlation_dict = correlation_dict
    correlation_dict = {}
    for i in range(16):
        correlation_dict[i] = OrderedDict()

    for cnd_key in tri_cnd.keys():

        # xyA -> abA
        if cnd_key[2] == 'A' and cnd_key[7] == 'A':
            if cnd_key in correlation_dict[0]:
                correlation_dict[0][cnd_key] += tri_cnd[cnd_key]
            else:
                correlation_dict[0][cnd_key] = tri_cnd[cnd_key]

        # xyA -> abC
        elif cnd_key[2] == 'A' and cnd_key[7] == 'C':
            if cnd_key in correlation_dict[1]:
                correlation_dict[1][cnd_key] += tri_cnd[cnd_key]
            else:
                correlation_dict[1][cnd_key] = tri_cnd[cnd_key]

        # xyA -> abG
        elif cnd_key[2] == 'A' and cnd_key[7] == 'G':
            if cnd_key in correlation_dict[2]:
                correlation_dict[2][cnd_key] += tri_cnd[cnd_key]
            else:
                correlation_dict[2][cnd_key] = tri_cnd[cnd_key]

        # xyA -> abT
        elif cnd_key[2] == 'A' and cnd_key[7] == 'T':
            if cnd_key in correlation_dict[3]:
                correlation_dict[3][cnd_key] += tri_cnd[cnd_key]
            else:
                correlation_dict[3][cnd_key] = tri_cnd[cnd_key]

        # xyC -> abA
        elif cnd_key[2] == 'C' and cnd_key[7] == 'A':
            if cnd_key in correlation_dict[4]:
                correlation_dict[4][cnd_key] += tri_cnd[cnd_key]
            else:
                correlation_dict[4][cnd_key] = tri_cnd[cnd_key]

        # xyC -> abC
        elif cnd_key[2] == 'C' and cnd_key[7] == 'C':
            if cnd_key in correlation_dict[5]:
                correlation_dict[5][cnd_key] += tri_cnd[cnd_key]
            else:
                correlation_dict[5][cnd_key] = tri_cnd[cnd_key]

        # xyC -> abG
        elif cnd_key[2] == 'C' and cnd_key[7] == 'G':
            if cnd_key in correlation_dict[6]:
                correlation_dict[6][cnd_key] += tri_cnd[cnd_key]
            else:
                correlation_dict[6][cnd_key] = tri_cnd[cnd_key]

        # xyC -> abT
        elif cnd_key[2] == 'C' and cnd_key[7] == 'T':
            if cnd_key in correlation_dict[7]:
                correlation_dict[7][cnd_key] += tri_cnd[cnd_key]
            else:
                correlation_dict[7][cnd_key] = tri_cnd[cnd_key]

        # xyG -> abA
        elif cnd_key[2] == 'G' and cnd_key[7] == 'A':
            if cnd_key in correlation_dict[8]:
                correlation_dict[8][cnd_key] += tri_cnd[cnd_key]
            else:
                correlation_dict[8][cnd_key] = tri_cnd[cnd_key]

        # xyG -> abC
        elif cnd_key[2] == 'G' and cnd_key[7] == 'C':
            if cnd_key in correlation_dict[9]:
                correlation_dict[9][cnd_key] += tri_cnd[cnd_key]
            else:
                correlation_dict[9][cnd_key] = tri_cnd[cnd_key]

        # xyG -> abG
        elif cnd_key[2] == 'G' and cnd_key[7] == 'G':
            if cnd_key in correlation_dict[10]:
                correlation_dict[10][cnd_key] += tri_cnd[cnd_key]
            else:
                correlation_dict[10][cnd_key] = tri_cnd[cnd_key]

        # xyG -> abT
        elif cnd_key[2] == 'G' and cnd_key[7] == 'T':
            if cnd_key in correlation_dict[11]:
                correlation_dict[11][cnd_key] += tri_cnd[cnd_key]
            else:
                correlation_dict[11][cnd_key] = tri_cnd[cnd_key]

        # xyT -> abA
        elif cnd_key[2] == 'T' and cnd_key[7] == 'A':
            if cnd_key in correlation_dict[12]:
                correlation_dict[12][cnd_key] += tri_cnd[cnd_key]
            else:
                correlation_dict[12][cnd_key] = tri_cnd[cnd_key]

        # xyT -> abC
        elif cnd_key[2] == 'T' and cnd_key[7] == 'C':
            if cnd_key in correlation_dict[13]:
                correlation_dict[13][cnd_key] += tri_cnd[cnd_key]
            else:
                correlation_dict[13][cnd_key] = tri_cnd[cnd_key]

        # xyT -> abG
        elif cnd_key[2] == 'T' and cnd_key[7] == 'G':
            if cnd_key in correlation_dict[14]:
                correlation_dict[14][cnd_key] += tri_cnd[cnd_key]
            else:
                correlation_dict[14][cnd_key] = tri_cnd[cnd_key]

        # xyT -> abT
        elif cnd_key[2] == 'T' and cnd_key[7] == 'T':
            if cnd_key in correlation_dict[15]:
                correlation_dict[15][cnd_key] += tri_cnd[cnd_key]
            else:
                correlation_dict[15][cnd_key] = tri_cnd[cnd_key]

    print(correlation_dict)

    # Make 16 plots, one for each substitution or conservation
    for i in range(16):

        real_list = []
        generated_list = []
        for ref_key in ref_correlation_dict[i]:
            for cnd_key in correlation_dict[i]:
                if ref_key == cnd_key:
                    real_list.append(ref_correlation_dict[i][ref_key])
                    generated_list.append(correlation_dict[i][cnd_key])
                    break

        print(str(i))
        print(real_list)
        print(generated_list)

        correlation = np.corrcoef(np.asarray(real_list), np.asarray(generated_list))
        print(correlation)

        a = np.polyfit(np.asarray(real_list), np.asarray(generated_list), 1)  # line of best fit
        yfit = [a[0] * x + a[1] for x in real_list]

        plt.figure(figsize=(12., 9.), dpi=300)
        plt.scatter(real_list, generated_list)
        plt.plot(real_list, yfit, c='#377eb8', linewidth=2)
        # plt.axis('square')
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xlabel('Real')
        plt.ylabel('Generated')
        plt.title(str(i) + ': r = ' + str(correlation[0, 1]))
        plt.savefig('cwgan_correlation_' + str(i) + '.png')
        plt.close()


if __name__ == "__main__":
    # main()
    extended_main()

    # correlation_test('generator_output_test_des_343.txt')
    # correlation_test('generator_output_test_des_1269_clean.txt')

