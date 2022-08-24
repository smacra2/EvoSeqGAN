import numpy as np
import os
import natsort


# Utility to extract all k-l divergence values generated from test in a given test file
def extract_values(directory):

    names = []
    for filename in os.scandir(directory):
        if filename.is_file() and filename.name.endswith('txt'):
            names.append(filename.name)

    # Sort alphanumerically and not lexicographically
    sortedList = natsort.natsorted(names)
    print(sortedList)
    print()

    # Extract time point from filename
    time_pts = []
    for name in sortedList:
        temp = name.split('_')[-1].split('.')[0]
        time_pts.append(int(temp))

    tri_list = []
    di_list = []
    single_list = []
    clip_val = 1.0
    for filename in sortedList:
        f = open(directory + '/' + filename, 'r')
        X = f.readlines()
        # print(X)
        values = []
        for line in X:
            if line.startswith("K-L Divergence:"):
                values.append(float(line[17:].rstrip()))
        f.close()
        print(values)
        for i in range(len(values)):
            if values[i] > clip_val:
                values[i] = clip_val
        tri_list.append(values[2])
        di_list.append(values[3])
        single_list.append(values[4])

    print()
    print('tri =', tri_list)
    print('di =', di_list)
    print('single =', single_list)
    print('time_pts =', time_pts)
    print()


# extract_values('Models/5a_1e-4_3e-4_100_32_1_48/')
# extract_values('Models/5c_bi_1e-4_3e-4_100_32_1_48/')


# extract_values('Models/6a_1e-4_5e-4_100_32_1_48/')
# extract_values('Models/6c_bi_1e-4_5e-4_100_32_1_48/')
extract_values('Models/5b_indels_1e-4_3e-4_100_32_1_48/')
