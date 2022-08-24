import pickle
import numpy as np
from itertools import product
from keras.models import load_model
from conditional_wgan_onlydescendant import define_generator
# from conditional_biwgan_onlydescendant import define_generator as define_bigenerator


# Helper function to display generator predicted alignments after decoding
def show_sequence(generator, condition, input_example):
    original_ancestor = condition
    ori_anc = np.chararray(len(original_ancestor))
    for j in range(len(original_ancestor)):
        anc_argmax = np.argmax(original_ancestor[j])

        if anc_argmax == 0:
            ori_anc[j] = 'A'
        elif anc_argmax == 1:
            ori_anc[j] = 'C'
        elif anc_argmax == 2:
            ori_anc[j] = 'G'
        elif anc_argmax == 3:
            ori_anc[j] = 'T'
        elif anc_argmax == 4:
            ori_anc[j] = '-'

    gen_align = generator.predict(input_example)
    # print(gen_align)
    # anc = np.chararray(len(gen_align[0]))
    des = np.chararray(len(gen_align[0]))
    for j in range(len(gen_align[0])):
        des_argmax = np.argmax(gen_align[0, j, :])

        if des_argmax == 0:
            des[j] = 'A'
        elif des_argmax == 1:
            des[j] = 'C'
        elif des_argmax == 2:
            des[j] = 'G'
        elif des_argmax == 3:
            des[j] = 'T'
        elif des_argmax == 4:
            des[j] = '-'

    fix_ori_anc = ''
    fix_des = ''
    for i in ori_anc:
        fix_ori_anc += str(i.decode('utf-8'))
    for i in des:
        fix_des += str(i.decode('utf-8'))
    # print("Condition Ancestral")
    # print(fix_ori_anc)
    # print("Generated Descendant")
    # print(fix_des)

    return fix_ori_anc, fix_des


def main(input_name, output_name, train_length=500, test_length=500, isBidirectional=0):

    if train_length == test_length:
        # Load the fitted model
        generator = load_model(input_name, compile=False)
    elif isBidirectional == 0:
        # Create new generator model using weights from old model, injected into new sequence length space.
        generator = define_generator(test_length, dropout=0.5)
        # for i in range(len(generator.get_weights())):
        #     print(generator.get_weights()[i].shape)
        # print(generator.get_weights())
        old_generator = load_model(input_name, compile=False)
        # for i in range(len(old_generator.get_weights())):
        #     print(old_generator.get_weights()[i].shape)
        # print(old_generator.get_weights())
        generator.set_weights(old_generator.get_weights())
    # elif isBidirectional == 1:
    #     # Create new bidirectional generator model using weights from old model, injected into new sequence length space.
    #     generator = define_bigenerator(test_length, dropout=0.5)
    #     old_generator = load_model(input_name, compile=False)
    #     generator.set_weights(old_generator.get_weights())

    LATENT_DIM = 10
    sequence_length = test_length  # change this value according to sequence_length of training data

    # # Create list of all sequence_length permutations to pass as conditional ancestor
    # alphabet = ['A', 'C', 'G', 'T']
    # # permutations = list(product(alphabet, repeat=sequence_length))  # too many to list once seq_length > 5
    #
    # # Select 1000 random 'permutations' manually for seq_length > 5
    # permutations = []
    # for i in range(1000):
    #     temp = ''
    #     for j in range(sequence_length):
    #         temp += str(np.random.choice(alphabet))
    #     permutations.append(temp)
    #
    # print("Number of all ancestor permutations: " + str(len(permutations)))
    #
    # conditional_ancestors = []
    # for i in range(len(permutations)):
    #     temp = ''.join(permutations[i])
    #     conditional_ancestors.append(temp)
    # print(conditional_ancestors)
    #
    # # Encode conditional ancestors in same way model was trained with
    # X = []
    # for i in range(len(conditional_ancestors)):
    #     ref = conditional_ancestors[i]
    #     # Initialize encoding sequence
    #     tmp = np.zeros((1, len(ref), 5))
    #     # Encode reference sequence characters
    #     for j in range(len(ref)):
    #         if ref[j] == 'A':
    #             tmp[0, j, 0] = 1
    #         elif ref[j] == 'C':
    #             tmp[0, j, 1] = 1
    #         elif ref[j] == 'G':
    #             tmp[0, j, 2] = 1
    #         elif ref[j] == 'T':
    #             tmp[0, j, 3] = 1
    #         else:
    #             tmp[0, j, 4] = 1
    #     X.append(tmp)
    # # print(X)

    # When test set is available, use it instead of randomly generating ancestors
    pickle_in = open('testData_gapless_15000', "rb")
    X = (pickle.load(pickle_in))  # encoded matched sequence pairs
    X = X[:-1]
    X = np.asarray(X)
    X = X[:, :, :, :5]  # ancestor is first 5 dim
    print("Number of all condition ancestors: ", len(X))
    pickle_in.close()

    # f = open('generator_output_test.csv', 'w+')
    f = open(output_name, 'w+')
    cond_anc = []
    map_des = []

    for i in range(len(X)):
        for j in range(3):  # test each condition ancestor just 3 times to limit size of output file
            noise = np.random.normal(size=(1, sequence_length, LATENT_DIM))
            # print(noise)
            condition = X[i]
            # print(condition)
            input_example = np.concatenate((noise, condition), axis=2)
            # print(input_example)
            cond, des = show_sequence(generator, condition[0], input_example)
            f.write(cond + '\n')
            f.write(des + '\n\n')
            # cond_anc.append(cond)
            # map_des.append(des)
        f.write('--------------------' + '\n\n')

    # f.write(str(cond_anc) + '\n')
    # f.write(str(map_des))

    f.close()


main('generator_intermediate_1389.h5', 'generator_output_test_des.txt', train_length=15000, test_length=15000)

# main('generator_intermediate_749.h5', 'generator_output_test_des.txt', train_length=5000, test_length=15000)
# main('generator_intermediate_19.h5', 'generator_output_test_des.txt', train_length=500, test_length=15000)

# main('generator_intermediate_749.h5', 'generator_output_test_des.txt', train_length=5000, test_length=15000, isBidirectional=1)
# main('generator_intermediate_19.h5', 'generator_output_test_des.txt', train_length=500, test_length=15000, isBidirectional=1)
