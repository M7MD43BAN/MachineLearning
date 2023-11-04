# Euclidean Distance represents the shortest distance between two points.
# It is the straight line distance between two points in Euclidean space.
# It is the most common distance metric used in Machine Learning.
# It is calculated as the square root of the sum of the squared differences between the two vectors.
# It is calculated as: d(x,y) = sqrt(sum((x_i - y_i)^2))

from scipy.spatial.distance import euclidean

point1 = (1, 2, 3)
point2 = (4, 5, 6)

euclidean_distance = euclidean(point1, point2)
print('Euclidean Distance between ', point1, ' and ', point2, ' is: ', euclidean_distance)
