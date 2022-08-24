# EvoSeqGAN

An LSTM-based WGAN-GP approach to simulating DNA sequence evolution. At the time of writing, EvoSeqGAN cannot use CuDNNLSTM due to lack of support for higher order gradients, so training is restricted to CPU. 

A project undertaken at McGill University within the context of an MSc degree. See thesis Generative Adversarial Networks for the Simulation of DNA Sequence Evolution.pdf for details. Paper to come later. 

# Data

The sequence alignment .maf files are available at http://repo.cs.mcgill.ca/PUB/blanchem/Boreoeutherian/. We used chr20.anc.maf for training.

The tree structure and nomenclature for species are available at http://hgdownload.cse.ucsc.edu/goldenpath/hg38/multiz100way/hg38.100way.nh. The ancestral sequences are labelled as (First character seq1)(First character seq2). For example, the most recent common ancestor of hg38 and panTro4 is \_HP.

See raw data file sample Real_Alignments_500.000_Lines.txt for an idea.

# Requirements

$ pip install -r dependencies.txt

# Data cleaning

Run clean_data.py (substitution only models) or clean_data_indels.py (indel models). Modify arguments such as input file names, sequence tags, chunk size, etc., as required. Creates realData and testData files with desired chunk size.

# Utilities

After cloning, make sure utility files decoder.py, decoder_indel.py, count_mutations.py, count_indels_mutation.py, evolution_test.py, extract_values_utlity.py, generator_output_test_onlydescendants.py, generator_indels_output_test_onlydescendants.py, and plot_metrics.py are in the main directory.

# Training

Run conditional_wgan_onlydescendant.py (substitution only models) or conditional_wgan_indels_onlydescendant.py (indel models). Modify arguments such as input file name (on line 134 or 319, respectively), hyperparameters (top of each file), etc., as required. Generator and critic models are saved periodically and checkpoint-style.

# Generating fake sequences

Requires a generator model file. Run generator_output_test_onlydescendants.py (substitution only models) or generator_indels_output_test_onlydescendants.py (indel models) with testData file. Modify arguments such as input generator model name, output file name, training length used, output test length desired, etc., as required.

To simulate over long time spans, run evolution_test.py (for substitution models only). Modify arguments such as original ancestral sequence, consecutive evolution count, independent test repeat count, etc., as required.

# Analyzing fake sequences

Requires probabilities analysis of realData file to use as reference. Run decoder.py (substitution only models) or decoder_indels.py (indel models). Modify input and output file names, as required. Creates extended_probabilities file.

For K-L divergence values and mutation probabilities analysis, run count_mutations.py (substitution only models) or count_mutations_indels.py (indel models) with extended_probabilities file. Use test output of a specific generator model and enter this file name on line 484 or 407, respectively. Copy/paste the function output to the top of each generator model test output file.

# Plotting metrics

For Wasserstein loss plotting, requires the saved console output of all training duration either whole or piecewise. Run the plot_loss function inside plot_metrics.py. Modify arguments such as input file path, as required.

For K-L divergence plotting, run extract_values_utility.py. Modify arguments such as input file path, as required. Save the output and copy as arguments to the plot_divergence function inside plot_metrics.py.