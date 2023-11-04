# Minkowski Distance is a generalized form of Euclidean and Manhattan Distance.
# It is calculated as: d(x,y) = (sum(|x_i - y_i|^p))^(1/p)

from scipy.spatial.distance import minkowski

point1 = (1, 2, 3)
point2 = (4, 5, 6)

minkowski_distance = minkowski(point1, point2, p=3)
print('Minkowski Distance between ', point1, ' and ', point2, ' is: ', minkowski_distance)
