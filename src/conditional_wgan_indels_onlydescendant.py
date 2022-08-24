import pickle
import numpy as np
# Disable GPU if model uses LSTM instead of GPU-optimized CuDNNLSTM
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Reshape, LSTM, CuDNNLSTM, Dropout, concatenate, Concatenate, TimeDistributed
from keras.layers.merge import _Merge
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from functools import partial
from decoder_indel import reconstruct


BATCH_SIZE = 64
NUM_EPOCHS = 1000000
TRAINING_RATIO = 3  # number of times critic is trained each time generator trains once
GRADIENT_PENALTY_WEIGHT = 100.0
LATENT_DIM = 10  # for sampling the generator
LEARNING_RATE = 0.000005  # for both generator and critic optimizers
NUM_HIDDEN = 48  # for both generator and critic models 


# Code based on https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py


# Computes Wasserstein loss for a sample batch
# Critic output is not a probability [0, 1] like in vanilla GAN, but a linear output
# to make EM distance between real and fake samples as large as possible
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


# Penalize gradient with respect to input samples, not critic weights
def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    # GRADIENT IS NONE: TO DEBUG (ERROR: CONVERT X TO TENSOR ERROR, NONE VALUES NOT SUPPORETD)
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)

    # Compute lambda * (1 - ||grad||)^2 for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


# Takes in latent-dim random noise and 5-dim ancestral sequence of sequence_length as condition
# Outputs 90-dim descendant sequence of sequence_length
def define_generator(sequence_length=100, dropout=0.3):
    model = Sequential()
    model.add(LSTM(NUM_HIDDEN, input_shape=(sequence_length, LATENT_DIM + 5), return_sequences=True))
    model.add(Dropout(dropout))
    # model.add(LSTM(400, return_sequences=False))
    # model.add(Dropout(dropout))
    model.add(TimeDistributed(Dense(90, activation='softmax')))
    model.add(Reshape((sequence_length, 90)))
    model._name = 'Generator'
    return model


# Takes in 90-dim descendant sequence of sequence_length and
# 5-dim ancestral sequence of sequence_length as condition (same ancestor as in generator)
# Output is NOT a probability but a linear output whose difference is maximized between 2 units
def define_critic(sequence_length=100, dropout=0.3):
    model = Sequential()
    model.add(LSTM(NUM_HIDDEN, input_shape=(sequence_length, 5 + 90), return_sequences=False))
    model.add(Dropout(dropout))
    # model.add(LSTM(400, return_sequences=False))
    # model.add(Dropout(dropout))
    model.add(Dense(1))
    model._name = 'Critic'
    return model


class RandomWeightedAverage(_Merge):
    # Select random points on the line between each pair of real and fake sequence inputs

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


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
    print("Condition Ancestral")
    print(fix_ori_anc)
    print("Generated Descendant")
    print(fix_des)

    anc_out, des_out = reconstruct(fix_ori_anc, fix_des)

    print("After reconstruction...")
    print("Condition Ancestral")
    print(anc_out)
    print("Generated Descendant")
    print(des_out)

    return anc_out, des_out


def main():
    # Read in cleaned/encoded data from pickle file
    pickle_in = open('realData_indels_50', "rb")
    X = np.asarray(pickle.load(pickle_in))  # encoded matched sequence pairs
    X = np.reshape(X, (X.shape[0], X.shape[2], X.shape[3]))
    print(X.shape)
    Y = np.asarray(pickle.load(pickle_in))  # associated label: 1 for real, 0 for fake
    print(Y.shape)
    pickle_in.close()

    sequence_length = len(X[0])  # should be 100bp, may experiment with 500bp later
    print(sequence_length)
    print()

    num_batches = int(X.shape[0] // BATCH_SIZE)  # 8,451 // BATCH_SIZE
    steps_per_epoch = int(num_batches // TRAINING_RATIO)  # a.k.a. number of times generator is actually trained
    dropout = 0.5
    # g_optimizer = RMSprop(0.000005)
    # c_optimizer = RMSprop(0.000005)
    g_optimizer = Adam(LEARNING_RATE, 0.0, 0.9)
    c_optimizer = Adam(LEARNING_RATE, 0.0, 0.9)

    # Initialize models for later compilation using Keras Model Functional API
    generator = define_generator(sequence_length, dropout)
    generator.summary()
    critic = define_critic(sequence_length, dropout)
    critic.summary()

    # Critic is not trainable when generator updates weights
    for layer in critic.layers:
        layer.trainable = False
    critic.trainable = False

    # Compile generator
    g_noise = Input(shape=(sequence_length, LATENT_DIM))
    g_condition = Input(shape=(sequence_length, 5))
    g_input = concatenate([g_noise, g_condition], axis=2)  # sample noise + 5-dim ancestor as condition
    print(K.int_shape(g_input))

    g_layers = generator(g_input)
    print(K.int_shape(g_layers))
    c_layers_for_g = critic(
        concatenate([g_layers, g_condition], axis=2))  # generator uses critic's loss to drive its updates
    print(K.int_shape(c_layers_for_g))

    generator_model = Model(inputs=[g_noise, g_condition], outputs=[c_layers_for_g], name='Generator_Model')
    generator_model.compile(optimizer=g_optimizer, loss=wasserstein_loss)
    print(generator_model.metrics_names)
    generator_model.summary()

    # Compile critic
    # Generator is not trainable when critic updates weights
    for layer in critic.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    critic.trainable = True
    generator.trainable = False

    # Critic takes original ancestor, real aligned sequences, and random noise as input.
    # Random noise and original ancestor is passed through generator to produce fake aligned sequences.
    # Do not concatenate real and fake aligned sequences to make compilation easier
    # (Keras complains when input and output sizes do not match)

    g_noise_for_c = Input(shape=(sequence_length, LATENT_DIM))
    g_condition = Input(shape=(sequence_length, 5))
    g_input_for_c = concatenate([g_noise_for_c, g_condition], axis=2)  # sample noise + 5-dim ancestor as condition
    g_samples_for_c = generator(g_input_for_c)
    print(K.int_shape(g_samples_for_c))

    g_alignments_condition = concatenate([g_samples_for_c, g_condition], axis=2)
    print(K.int_shape(g_alignments_condition))
    c_output_from_g = critic(g_alignments_condition)
    print(K.int_shape(c_output_from_g))

    real_descendants = Input(shape=(sequence_length, 90))
    real_condition = Input(shape=(sequence_length, 5))
    real_descendants_condition = concatenate([real_descendants, real_condition],
                                             axis=2)  # aligned sequences + 5-dim ancestor as condition
    print(K.int_shape(real_descendants_condition))
    c_output_from_real_descendants = critic(real_descendants_condition)
    print(K.int_shape(c_output_from_real_descendants))

    # Also generate weighted-averages of real and fake alignments for gradient norm penalty
    averaged_samples = RandomWeightedAverage()([real_descendants_condition, g_alignments_condition])
    print(K.int_shape(averaged_samples))

    # Run averaged samples through critic. Note we do not use critic output here;
    # Only concerned with gradient norm for penalty computation
    averaged_samples_out = critic(averaged_samples)
    print(K.int_shape(averaged_samples_out))

    # Use partial() to get around Keras loss function limitation of requiring only two arguments
    partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=averaged_samples,
                              gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
    partial_gp_loss.__name__ = 'gradient_penalty'

    critic_model = Model(inputs=[real_descendants, real_condition, g_noise_for_c, g_condition],
                         outputs=[c_output_from_real_descendants, c_output_from_g, averaged_samples_out],
                         name='Critic_Model')
    critic_model.compile(optimizer=c_optimizer, loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss])
    print(critic_model.metrics_names)
    critic_model.summary()

    # Make three labels for training: y_positive (-1) for real samples, y_negative (1) for fake samples,
    # y_dummy for averaged_samples for gradient penalty and is unused
    y_positive = -np.ones((BATCH_SIZE, 1), dtype=np.float32)
    y_negative = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    y_dummy = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

    critic_loss_hist_real = []
    critic_loss_hist_fake = []
    generator_loss_hist_fake = []
    sequence_output_hist = []
    stable_condition_ancestor = X[-1, :, :5].copy().reshape(1, sequence_length, 5)
    # print(stable_condition_ancestor)

    best_loss_so_far = [99999999, 0]  # init to large initial loss, and 0 batches since last update
    # Training step
    for epoch in range(NUM_EPOCHS):
        # if best_loss_so_far[1] > 20000 * num_batches:  # if it's been 20000 * TRAINING_RATIO epochs since last improvement in loss
        #     break

        np.random.shuffle(X)
        print("Epoch: ", epoch, " of ", NUM_EPOCHS)
        print("Number of times critic is trained per epoch: ", steps_per_epoch * TRAINING_RATIO)
        print("Number of times generator is trained per epoch: ", steps_per_epoch)
        critic_batch_counter = 0

        if epoch % 50 == 24:
            critic.save("critic_intermediate_%d.h5" % epoch)
            generator.save("generator_intermediate_%d.h5" % epoch)

        for batch in range(steps_per_epoch):
            start_index = critic_batch_counter * BATCH_SIZE * TRAINING_RATIO
            end_index = start_index + (BATCH_SIZE * TRAINING_RATIO)

            critic_real_alignments = X[start_index:end_index]
            # print(K.int_shape(critic_real_alignments))
            critic_real_ancestor = critic_real_alignments[:, :, :5]  # ancestor is first 5 dimensions of alignment
            # print(K.int_shape(critic_real_ancestor))
            critic_real_descendant = critic_real_alignments[:, :, 5:]  # descendant is remaining dimensions of alignment
            critic_batch_counter += 1

            # First train critic TRAINING_RATIO more times than generator
            c_loss_to_average_real = []
            c_loss_to_average_fake = []
            c_loss_to_average_grad_penalty = []
            total_c_loss = []
            for i in range(TRAINING_RATIO):
                real_descendant_batch = critic_real_descendant[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                real_ancestor_batch = critic_real_ancestor[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]

                random_noise = np.random.normal(size=(BATCH_SIZE, sequence_length, LATENT_DIM))
                generator_ancestor = real_ancestor_batch[::-1]  # take from the back of real ancestors

                c_loss = critic_model.train_on_batch([real_descendant_batch, real_ancestor_batch,
                                                      random_noise, generator_ancestor],
                                                     [y_positive, y_negative, y_dummy])

                c_loss_to_average_real.append(c_loss[1])
                c_loss_to_average_fake.append(c_loss[2])
                c_loss_to_average_grad_penalty.append(c_loss[3])
                total_c_loss.append(c_loss[0])
            avg_real = np.mean(c_loss_to_average_real)
            avg_fake = np.mean(c_loss_to_average_fake)
            critic_loss_hist_real.append(avg_real)
            critic_loss_hist_fake.append(avg_fake)
            avg_grad_penalty = np.mean(c_loss_to_average_grad_penalty)
            avg_total = np.mean(total_c_loss)

            # Then train generator once
            random_noise = np.random.normal(size=(BATCH_SIZE, sequence_length, LATENT_DIM))
            generator_ancestor = critic_real_ancestor[:-BATCH_SIZE - 1:-1]
            g_loss = generator_model.train_on_batch([random_noise, generator_ancestor], y_positive)
            generator_loss_hist_fake.append(g_loss)

            w_loss = avg_real + avg_fake  # real loss already includes the negative sign
            if abs(w_loss) <= best_loss_so_far[0] and epoch > 1:
                best_loss_so_far[0] = abs(w_loss)
                best_loss_so_far[1] = 0
                critic.save("critic_checkpoint_%.10f_%d_%d.h5" % (w_loss, epoch, batch))
                generator.save("generator_checkpoint_%.10f_%d_%d.h5" % (w_loss, epoch, batch))
            else:
                best_loss_so_far[1] += 1

            # Display some progress
            if batch % 25 == 24:
                print(f'Epoch: {epoch} \t Batch: {batch} \t '
                      f'Total Loss To Minimize: {avg_total} \t Wasserstein Loss: {w_loss} \t '
                      f'Critic Loss Real: {avg_real} \t Critic Loss on Fake: {avg_fake} \t '
                      f'Gradient Penalty: {avg_grad_penalty} \t Generator Loss: {g_loss}')

            # Show some sample alignments with random noise but consistent conditional ancestor
            if batch % 50 == 24:
                noise = np.random.normal(size=(1, sequence_length, LATENT_DIM))
                # print(noise)
                condition = stable_condition_ancestor
                # print(condition)
                input_example = np.concatenate((noise, condition), axis=2)
                # print(input_example)
                cond, des = show_sequence(generator, condition[0], input_example)
                sequence_output_hist.append(cond)
                sequence_output_hist.append(des)

    hist = np.vstack((critic_loss_hist_real, critic_loss_hist_fake, generator_loss_hist_fake)).T

    # Save the organized history into a txt file
    np.savetxt("cwgan_loss_history.txt", hist, delimiter=",")

    # Save history of generator outputs to assess diversity and quality
    f = open('cwgan_generated_sequences.txt', 'a+')
    for i in range(len(sequence_output_hist)):
        f.write(sequence_output_hist[i] + '\n')
        # if i % 3 == 2:
        #     f.write('\n')
        if i % 2 == 1:
            f.write('\n')
    f.close()

    # Save final models in full tensorflow savedModel format for later loading if needed
    critic.save('critic')
    generator.save('generator')
    critic_model.save('cwgan_critic')
    generator_model.save('cwgan_generator')


if __name__ == "__main__":
    main()
