# Hamming Distance measures the similarity between two strings of the same length.
"""
The Hamming distance between two strings of equal length is
the number of positions at which the corresponding symbols are different.
"""

from scipy.spatial.distance import hamming

string1 = 'euclidean'
string2 = 'manhattan'

# len(string1) == len(string2) is a must
# len(string1) for the number of items in the array that are different, not the proportion
hamming_distance = hamming(list(string1), list(string2))*len(string1)
print('Hamming Distance between ', string1, ' and ', string2, ' is: ', hamming_distance)
