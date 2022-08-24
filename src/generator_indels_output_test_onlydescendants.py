import pickle
import numpy as np
from itertools import product
from keras.models import load_model
from conditional_wgan_indels_onlydescendant import define_generator, reconstruct


# Helper function to display generator predicted alignments after decoding
def show_sequence(generator, condition, input_example):
    original_ancestor = condition
    ori_anc = []
    for j in range(len(original_ancestor)):
        anc_argmax = np.argmax(original_ancestor[j])

        if anc_argmax == 0:
            ori_anc.append('A')
        elif anc_argmax == 1:
            ori_anc.append('C')
        elif anc_argmax == 2:
            ori_anc.append('G')
        elif anc_argmax == 3:
            ori_anc.append('T')
        elif anc_argmax == 4:
            ori_anc.append('-')

    gen_align = generator.predict(input_example)
    # print(gen_align)
    des = []
    for j in range(len(gen_align[0])):
        des_argmax = np.argmax(gen_align[0, j, :])

        if des_argmax == 0:
            des.append('A')
        elif des_argmax == 1:
            des.append('C')
        elif des_argmax == 2:
            des.append('G')
        elif des_argmax == 3:
            des.append('T')
        elif des_argmax == 4:
            des.append('-')

        elif des_argmax == 5:
            des.append('iAAn')
        elif des_argmax == 6:
            des.append('iACn')
        elif des_argmax == 7:
            des.append('iAGn')
        elif des_argmax == 8:
            des.append('iATn')
        elif des_argmax == 9:
            des.append('iCAn')
        elif des_argmax == 10:
            des.append('iCCn')
        elif des_argmax == 11:
            des.append('iCGn')
        elif des_argmax == 12:
            des.append('iCTn')
        elif des_argmax == 13:
            des.append('iGAn')
        elif des_argmax == 14:
            des.append('iGCn')
        elif des_argmax == 15:
            des.append('iGGn')
        elif des_argmax == 16:
            des.append('iGTn')
        elif des_argmax == 17:
            des.append('iTAn')
        elif des_argmax == 18:
            des.append('iTCn')
        elif des_argmax == 19:
            des.append('iTGn')
        elif des_argmax == 20:
            des.append('iTTn')

        elif des_argmax == 21:
            des.append('iAAAn')
        elif des_argmax == 22:
            des.append('iAACn')
        elif des_argmax == 23:
            des.append('iAAGn')
        elif des_argmax == 24:
            des.append('iAATn')
        elif des_argmax == 25:
            des.append('iACAn')
        elif des_argmax == 26:
            des.append('iACCn')
        elif des_argmax == 27:
            des.append('iACGn')
        elif des_argmax == 28:
            des.append('iACTn')
        elif des_argmax == 29:
            des.append('iAGAn')
        elif des_argmax == 30:
            des.append('iAGCn')
        elif des_argmax == 31:
            des.append('iAGGn')
        elif des_argmax == 32:
            des.append('iAGTn')
        elif des_argmax == 33:
            des.append('iATAn')
        elif des_argmax == 34:
            des.append('iATCn')
        elif des_argmax == 35:
            des.append('iATGn')
        elif des_argmax == 36:
            des.append('iATTn')

        elif des_argmax == 37:
            des.append('iCAAn')
        elif des_argmax == 38:
            des.append('iCACn')
        elif des_argmax == 39:
            des.append('iCAGn')
        elif des_argmax == 40:
            des.append('iCATn')
        elif des_argmax == 41:
            des.append('iCCAn')
        elif des_argmax == 42:
            des.append('iCCCn')
        elif des_argmax == 43:
            des.append('iCCGn')
        elif des_argmax == 44:
            des.append('iCCTn')
        elif des_argmax == 45:
            des.append('iCGAn')
        elif des_argmax == 46:
            des.append('iCGCn')
        elif des_argmax == 47:
            des.append('iCGGn')
        elif des_argmax == 48:
            des.append('iCGTn')
        elif des_argmax == 49:
            des.append('iCTAn')
        elif des_argmax == 50:
            des.append('iCTCn')
        elif des_argmax == 51:
            des.append('iCTGn')
        elif des_argmax == 52:
            des.append('iCTTn')

        elif des_argmax == 53:
            des.append('iGAAn')
        elif des_argmax == 54:
            des.append('iGACn')
        elif des_argmax == 55:
            des.append('iGAGn')
        elif des_argmax == 56:
            des.append('iGATn')
        elif des_argmax == 57:
            des.append('iGCAn')
        elif des_argmax == 58:
            des.append('iGCCn')
        elif des_argmax == 59:
            des.append('iGCGn')
        elif des_argmax == 60:
            des.append('iGCTn')
        elif des_argmax == 61:
            des.append('iGGAn')
        elif des_argmax == 62:
            des.append('iGGCn')
        elif des_argmax == 63:
            des.append('iGGGn')
        elif des_argmax == 64:
            des.append('iGGTn')
        elif des_argmax == 65:
            des.append('iGTAn')
        elif des_argmax == 66:
            des.append('iGTCn')
        elif des_argmax == 67:
            des.append('iGTGn')
        elif des_argmax == 68:
            des.append('iGTTn')

        elif des_argmax == 69:
            des.append('iTAAn')
        elif des_argmax == 70:
            des.append('iTACn')
        elif des_argmax == 71:
            des.append('iTAGn')
        elif des_argmax == 72:
            des.append('iTATn')
        elif des_argmax == 73:
            des.append('iTCAn')
        elif des_argmax == 74:
            des.append('iTCCn')
        elif des_argmax == 75:
            des.append('iTCGn')
        elif des_argmax == 76:
            des.append('iTCTn')
        elif des_argmax == 77:
            des.append('iTGAn')
        elif des_argmax == 78:
            des.append('iTGCn')
        elif des_argmax == 79:
            des.append('iTGGn')
        elif des_argmax == 80:
            des.append('iTGTn')
        elif des_argmax == 81:
            des.append('iTTAn')
        elif des_argmax == 82:
            des.append('iTTCn')
        elif des_argmax == 83:
            des.append('iTTGn')
        elif des_argmax == 84:
            des.append('iTTTn')

        elif des_argmax == 85:
            des.append('iAXn')
        elif des_argmax == 86:
            des.append('iCXn')
        elif des_argmax == 87:
            des.append('iGXn')
        elif des_argmax == 88:
            des.append('iTXn')

        elif des_argmax == 89:
            des.append('Z')  # dummy

    fix_ori_anc = ''
    fix_des = ''
    for i in ori_anc:
        fix_ori_anc += str(i)
    for i in des:
        fix_des += str(i)
    # print("Condition Ancestral")
    # print(fix_ori_anc)
    # print("Generated Descendant")
    # print(fix_des)

    anc_out, des_out = reconstruct(fix_ori_anc, fix_des)

    # print("After reconstruction...")
    # print("Condition Ancestral")
    # print(anc_out)
    # print("Generated Descendant")
    # print(des_out)

    return anc_out, des_out


def main(input_name, output_name, train_length=500, test_length=500):

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
    pickle_in = open('testData_indels_1500', "rb")
    X = np.asarray(pickle.load(pickle_in))  # encoded matched sequence pairs
    # X = np.reshape(X, (X.shape[0], X.shape[2], X.shape[3]))
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


main('generator_intermediate_2539.h5', 'generator_indel_output_test_des.txt', train_length=3000, test_length=1500)

# main('generator_intermediate_299.h5', 'generator_indel_output_test_des.txt', train_length=500, test_length=1500)
# main('generator_intermediate_19.h5', 'generator_indel_output_test_des.txt', train_length=50, test_length=1500)
