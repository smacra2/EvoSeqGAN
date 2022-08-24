import math
import pickle
from itertools import product
import numpy as np


def count_mutations(file_name, mutation_val=5, deletion_val=5, insertion_val=5):
    f = open(file_name)
    data = []
    for row in f:
        if row[0] != '\n' and row != '--------------------\n':
            data.append(row.rstrip())
    print(len(data))
    # for i in range(len(data)):
    #     print(data[i])

    mutations_list = []
    insertions_list = []
    deletions_list = []
    conservation_types = {}
    mutation_types = {}
    for j in range(0, len(data), 2):
        anc = data[j]
        des = data[j + 1]

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

        # if num_mutations > mutation_val:
        #     print("MUTATION")
        # if num_deletions > deletion_val:
        #     print("DELETION")
        # if num_insertions > insertion_val:
        #     print("INSERTION")
        # if num_mutations > 0 or num_insertions > 0 or num_deletions > 0:
        #     print(anc + '\n' + des)

        mutations_list.append(num_mutations)
        insertions_list.append(num_insertions)
        deletions_list.append(num_deletions)

    print(mutations_list)
    print(insertions_list)
    print(deletions_list)

    print("Number of sequences compared: ", len(mutations_list))
    print("Max number of mutations in alignment: ", np.max(mutations_list))
    print("Mean number of mutations in alignment: ", np.mean(mutations_list))
    print("Median number of mutations in alignment: ", np.median(mutations_list))

    print("Max number of insertions in alignment: ", np.max(insertions_list))
    print("Mean number of insertions in alignment: ", np.mean(insertions_list))
    print("Median number of insertions in alignment: ", np.median(insertions_list))

    print("Max number of deletions in alignment: ", np.max(deletions_list))
    print("Mean number of deletions in alignment: ", np.mean(deletions_list))
    print("Median number of deletions in alignment: ", np.median(deletions_list))

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
    probabilities, conditional_probabilities = count_mutations('generator_indel_output_test_des.txt', mutation_val=40,
                                                               deletion_val=12, insertion_val=6)

    indel_reference = {'G->G': 0.2054911822523275, 'T->T': 0.27636391545576244, 'C->C': 0.20439245909242262,
                       'A->A': 0.2742263718361839, 'G->A': 0.003981774603454654, 'A->G': 0.005505198548201901,
                       'A->T': 0.0008561874224453078, 'C->G': 0.0010380014774439424, 'T->A': 0.0008802888994914589,
                       'T->C': 0.0055226311699488745, 'G->T': 0.0010803545584667903, 'T->G': 0.001058827025571199,
                       'A->C': 0.0010411604089014476, 'C->T': 0.003862554190299179, '-->C': 0.0008436686940766759,
                       'C->A': 0.0010912353223759752, '-->T': 0.00125514876578208, '-->A': 0.001252340826708742,
                       'T->-': 0.0026550233913024556, 'A->-': 0.002630804916794915, '-->G': 0.0007079516388653399,
                       'C->-': 0.0015933884266579185, 'G->-': 0.0016135119900168406, 'G->C': 0.0010560190864978612}

    indel_condition_reference = {'G->G': 0.963739062156199, 'T->T': 0.96468602952047, 'C->C': 0.9642170774704246,
                                 'A->A': 0.9647035774686031, 'G->A': 0.018674240325011783, 'A->G': 0.019366790650236682,
                                 'A->T': 0.0030119899260090967, 'C->G': 0.004896749887129694,
                                 'T->A': 0.003072768750872946,
                                 'T->C': 0.01927749911582345, 'G->T': 0.00506678621223985, 'T->G': 0.003695980488490186,
                                 'A->C': 0.0036627081650116087, 'C->T': 0.01822151721975876,
                                 '-->C': 0.20784573701504583,
                                 'C->A': 0.005147879418085963, '-->T': 0.30921773217271, '-->A': 0.3085259699083415,
                                 'T->-': 0.009267722124343403, 'A->-': 0.009254933790139457, '-->G': 0.1744105609039027,
                                 'C->-': 0.007516776004600914, 'G->-': 0.0075672567308858315,
                                 'G->C': 0.004952654575663513}

    # For convenience
    # -->A: 0.001252340826708742
    # -->C: 0.0008436686940766759
    # -->G: 0.0007079516388653399
    # -->T: 0.00125514876578208
    # A->-: 0.002630804916794915
    # A->A: 0.2742263718361839
    # A->C: 0.0010411604089014476
    # A->G: 0.005505198548201901
    # A->T: 0.0008561874224453078
    # C->-: 0.0015933884266579185
    # C->A: 0.0010912353223759752
    # C->C: 0.20439245909242262
    # C->G: 0.0010380014774439424
    # C->T: 0.003862554190299179
    # G->-: 0.0016135119900168406
    # G->A: 0.003981774603454654
    # G->C: 0.0010560190864978612
    # G->G: 0.2054911822523275
    # G->T: 0.0010803545584667903
    # T->-: 0.0026550233913024556
    # T->A: 0.0008802888994914589
    # T->C: 0.0055226311699488745
    # T->G: 0.001058827025571199
    # T->T: 0.27636391545576244

    # Conditional probabilities on ancestor character
    # -->A: 0.3085259699083415
    # -->C: 0.20784573701504583
    # -->G: 0.1744105609039027
    # -->T: 0.30921773217271
    # A->-: 0.009254933790139457
    # A->A: 0.9647035774686031
    # A->C: 0.0036627081650116087
    # A->G: 0.019366790650236682
    # A->T: 0.0030119899260090967
    # C->-: 0.007516776004600914
    # C->A: 0.005147879418085963
    # C->C: 0.9642170774704246
    # C->G: 0.004896749887129694
    # C->T: 0.01822151721975876
    # G->-: 0.0075672567308858315
    # G->A: 0.018674240325011783
    # G->C: 0.004952654575663513
    # G->G: 0.963739062156199
    # G->T: 0.00506678621223985
    # T->-: 0.009267722124343403
    # T->A: 0.003072768750872946
    # T->C: 0.01927749911582345
    # T->G: 0.003695980488490186
    # T->T: 0.96468602952047

    for key in sorted(probabilities.keys()):
        print(key, " : ", probabilities[key])

    # Make sure every substitution in reference dictionary is in generated dictionary, even with probability 0
    for key in indel_reference:
        if key not in probabilities:
            probabilities[key] = 0
            print("Missing key: ", key)

    euclidean_distance = 0
    for key in probabilities:
        for key2 in indel_reference:
            if key == key2:
                euclidean_distance += (indel_reference[key] - probabilities[key]) ** 2
    print("Squared Euclidean Distance: ", euclidean_distance)
    euclidean_distance = math.sqrt(euclidean_distance)
    print("Euclidean Distance: ", euclidean_distance)

    kl_divergence = 0
    for key in probabilities:
        for key2 in indel_reference:
            if key == key2:
                temp = math.log((probabilities[key] / indel_reference[key]) + 1e-8) * probabilities[key]
                kl_divergence += temp
    print("K-L Divergence: ", kl_divergence)

    print("\nNow repeating for conditional probabilities....")

    for key in sorted(conditional_probabilities.keys()):
        print(key, " : ", conditional_probabilities[key])

    for key in indel_condition_reference:
        if key not in conditional_probabilities:
            conditional_probabilities[key] = 0
            print("Missing key: ", key)

    euclidean_distance = 0
    for key in conditional_probabilities:
        for key2 in indel_condition_reference:
            if key == key2:
                euclidean_distance += (indel_condition_reference[key] - conditional_probabilities[key]) ** 2
    print("Squared Euclidean Distance: ", euclidean_distance)
    euclidean_distance = math.sqrt(euclidean_distance)
    print("Euclidean Distance: ", euclidean_distance)

    kl_divergence = 0
    for key in conditional_probabilities:
        for key2 in indel_condition_reference:
            if key == key2:
                temp = math.log((conditional_probabilities[key] / indel_condition_reference[key]) + 1e-8) * conditional_probabilities[key]
                kl_divergence += temp
    print("K-L Divergence: ", kl_divergence)


def count_extended_mutations(file_name):
    f = open(file_name)
    data = []
    for row in f:
        if row[0] != '\n' and row != '--------------------\n':
            data.append(row.rstrip())
    print(len(data))

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
    pickle_in = open('realData_indels_1500_extended_probabilities', "rb")
    # Load saved mutation probabilities from reference data
    tri_ref = pickle.load(pickle_in)
    di_ref = pickle.load(pickle_in)
    single_ref = pickle.load(pickle_in)
    pickle_in.close()

    tri_cnd, di_cnd, single_cnd = count_extended_mutations('generator_indel_output_test_des.txt')

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


if __name__ == "__main__":
    main()
    extended_main()
