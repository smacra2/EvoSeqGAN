# Simulation-of-DNA-Sequence-Evolution

An undergraduate research project undertaken at McGill University.

GAN-based approach to simulating sequence evolution:

Given two datasets: 

(1) containing real sequence pairs consisting of a modern human DNA sequence aligned with what experts 
 agree is its real ancestral sequence, and
                    
(2) containing fake sequence pairs consisting of a modern human DNA sequence aligned with the output of 
EVOLVER, a hardcoded evolution simulator.

1. A discriminator model that assigns a binary real/fake label to a given input sequence pair that has been trained 
   on the above two datasets. 
   
   a) Find data cleaning, model training, testing, and plotting code in src folder.
   
   b) Find plots and diagrams in root.

2. (To come) A generator model that creates candidate sequence pairs.

3. (To come) A generative adversarial network that trains the two models above in concert, such that the generator
    ultimately creates seemingly real sequence pairs (i.e. sequence pairs undistinguishable from the real dataset).
