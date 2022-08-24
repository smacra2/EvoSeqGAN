import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from conditional_wgan_onlydescendant import define_generator


# Helper function to encode a user-created ancestral sequence
def encode_ancestor(ancestor):
    encoded_ancestor = np.zeros((1, len(ancestor), 5))

    for i in range(len(ancestor)):
        if ancestor[i] == 'A':
            encoded_ancestor[0, i, 0] = 1
        elif ancestor[i] == 'C':
            encoded_ancestor[0, i, 1] = 1
        elif ancestor[i] == 'G':
            encoded_ancestor[0, i, 2] = 1
        elif ancestor[i] == 'T':
            encoded_ancestor[0, i, 3] = 1
        elif ancestor[i] == '-':
            encoded_ancestor[0, i, 4] = 1

    return encoded_ancestor


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

    return fix_ori_anc, fix_des


# Evolve once independently test_number of times
def diversity_test(input_name, ancestor, test_number=1000, train_length=500, test_length=500):

    if train_length == test_length:
        # Load the fitted model
        generator = load_model(input_name, compile=False)
    else:
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

    LATENT_DIM = 10
    sequence_length = test_length  # change this value according to sequence_length of training data
    anc_data = []
    des_data = []
    for i in range(test_length):
        anc_data.append([])
        des_data.append([])

    for i in range(test_number):
        noise = np.random.normal(size=(1, sequence_length, LATENT_DIM))
        # print(noise)
        condition = ancestor
        # print(condition)
        input_example = np.concatenate((noise, condition), axis=2)
        # print(input_example)
        cond, des = show_sequence(generator, condition[0], input_example)
        # print(cond)
        print(des)
        for j in range(test_length):
            anc_data[j].append(cond[j])
            des_data[j].append(des[j])

    # num_A = {}
    # num_C = {}
    # num_G = {}
    # num_T = {}
    # for i in range(test_length):
    #     num_A[i] = 0
    #     num_C[i] = 0
    #     num_G[i] = 0
    #     num_T[i] = 0
    #
    #     for j in range(test_number):
    #         if des_data[i][j] == 'A':
    #             num_A[i] += 1
    #         elif des_data[i][j] == 'C':
    #             num_C[i] += 1
    #         elif des_data[i][j] == 'G':
    #             num_G[i] += 1
    #         elif des_data[i][j] == 'T':
    #             num_T[i] += 1
    #
    # print(num_A)
    # print(num_C)
    # print(num_G)
    # print(num_T)
    #
    # plt.figure(figsize=(12., 9.), dpi=300)
    # plt.bar(num_A.keys(), num_A.values(), color='crimson', width=0.35, alpha=0.6)
    # plt.bar(num_C.keys(), num_C.values(), color='darkcyan', width=0.35, alpha=0.6)
    # plt.bar(num_G.keys(), num_G.values(), color='blueviolet', width=0.35, alpha=0.6)
    # plt.bar(num_T.keys(), num_T.values(), color='chocolate', width=0.35, alpha=0.6)
    #
    # plt.xlabel('position')
    # plt.ylabel('frequency')
    # plt.legend(['A', 'C', 'G', 'T'])
    # plt.title('frequency versus position')
    # plt.savefig('cwgan_diversity_test_des.png')
    # plt.close()
    #
    # num_A = {}
    # num_C = {}
    # num_G = {}
    # num_T = {}
    # for i in range(test_length):
    #     num_A[i] = 0
    #     num_C[i] = 0
    #     num_G[i] = 0
    #     num_T[i] = 0
    #
    #     for j in range(test_number):
    #         if anc_data[i][j] == 'A':
    #             num_A[i] += 1
    #         elif anc_data[i][j] == 'C':
    #             num_C[i] += 1
    #         elif anc_data[i][j] == 'G':
    #             num_G[i] += 1
    #         elif anc_data[i][j] == 'T':
    #             num_T[i] += 1
    #
    # print(num_A)
    # print(num_C)
    # print(num_G)
    # print(num_T)
    #
    # plt.figure(figsize=(12., 9.), dpi=300)
    # plt.bar(num_A.keys(), num_A.values(), color='crimson', width=0.35, alpha=0.6)
    # plt.bar(num_C.keys(), num_C.values(), color='darkcyan', width=0.35, alpha=0.6)
    # plt.bar(num_G.keys(), num_G.values(), color='blueviolet', width=0.35, alpha=0.6)
    # plt.bar(num_T.keys(), num_T.values(), color='chocolate', width=0.35, alpha=0.6)
    #
    # plt.xlabel('position')
    # plt.ylabel('frequency')
    # plt.legend(['A', 'C', 'G', 'T'])
    # plt.title('frequency versus position')
    # plt.savefig('cwgan_diversity_test_anc.png')
    # plt.close()


def sequential_evolution_test(input_name, ancestor, consecutive_number=1000, repeat_test=100,
                              train_length=500, test_length=500):

    if train_length == test_length:
        # Load the fitted model
        generator = load_model(input_name, compile=False)
    else:
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

    LATENT_DIM = 10
    sequence_length = test_length  # change this value according to sequence_length of training data
    anc_data = []
    des_data = []
    for i in range(test_length):
        anc_data.append([])
        des_data.append([])

    for iteration in range(repeat_test):
        condition = ancestor
        des = ''
        for i in range(consecutive_number):
            noise = np.random.normal(size=(1, sequence_length, LATENT_DIM))
            # print(noise)
            input_example = np.concatenate((noise, condition), axis=2)
            # print(input_example)
            cond, new_des = show_sequence(generator, condition[0], input_example)
            condition = encode_ancestor(new_des)
            des = new_des

            # if i == 0:
            #     print(cond)
            # print(des)
            # if cond != des:
            #     print(str(i) + ' ' + des)
            # for j in range(test_length):
            #     des_data[j].append(des[j])

        print(des)


if __name__ == "__main__":

    ancestor_seq = 'AAAACCCCGGGGTTTTAAAACCCCGGGGTTTTCGAAAACCCCGGGGTTTTAAAACCCCGGGGCGTTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTT'
    # ancestor_seq = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'

    # Train set first and last
    # ancestor_seq = 'CTTGATGTGTGTTTGATATGTTTTAAAGTTTTGTATGGCTCCTTCATGACTCAATTTGTTTTCCTTCCCCTAGTCCCCTTCACTCTTCCTTCCATATTTT'
    # ancestor_seq = 'TGGTTAGATTACAGCTTGGGATATGACGGATGCACCTCTGAGCATAATCTCCATGAGATAACCCTAACCAAAACTTTTTGTCCAGTTCTTCTGTGGCATG'

    # Test set
    # ancestor_seq = 'AACACTGATGTACACGCACTGTTTTTTAGTGAGATGGCCTGGCAGTGCCTGGAAGTTGGCCTCCTGGCTTCTGTCAGCAGGAATCAGAGATGAAGACCCT'

    # diversity_test('generator_intermediate_343.h5', encode_ancestor(ancestor_seq), test_number=1000,
    #                train_length=15000, test_length=len(ancestor_seq))

    # sequential_evolution_test('generator_intermediate_869.h5', encode_ancestor(ancestor_seq), consecutive_number=100,
    #                           repeat_test=1000, train_length=15000, test_length=len(ancestor_seq))

    sequential_evolution_test('generator_intermediate_1269_unisub.h5', encode_ancestor(ancestor_seq), consecutive_number=100,
                              repeat_test=1000, train_length=15000, test_length=len(ancestor_seq))
