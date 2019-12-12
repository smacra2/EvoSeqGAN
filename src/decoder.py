import pickle
import numpy as np

pickle_in = open('cleanData', "rb")
realMatches = pickle.load(pickle_in)  # real matched sequence pairs
fakeMatches = pickle.load(pickle_in)  # fake matched sequence pairs
X = pickle.load(pickle_in)  # encoded matched sequence pairs
Y = pickle.load(pickle_in)  # associated label: 1 for real, 0 for fake
pickle_in.close()


# Function to decode the two-hot encoding of the cleaned data
# For viewing of sequence prediction results
def decode(sequence):
    if sequence not in range(len(X)):
        raise ValueError("Invalid index")
    else:
        tmp = X[sequence]
        ref = np.chararray(len(tmp[0]))
        cnd = np.chararray(len(tmp[0]))

        for j in range(len(tmp[0])):
            if tmp[0][j][0] == 1:
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


ref = []
cnd = []

ref, cnd = decode(120223)

fix_ref = ''
fix_cnd = ''

for i in ref:
    fix_ref += str(i.decode('utf-8'))
for i in cnd:
    fix_cnd += str(i.decode('utf-8'))

print("Ancestral")
print(fix_ref)
print("\nDescendant")
print(fix_cnd)
