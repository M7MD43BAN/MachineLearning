# Manhattan Distance is the sum of absolute differences between points across all the dimensions.
# The distance between two points measured along axes at right angles.
# It is calculated as: d(x,y) = sum(|x_i - y_i|)

from scipy.spatial.distance import cityblock

point1 = (1, 2, 3)
point2 = (4, 5, 6)

manhattan_distance = cityblock(point1, point2)
print('Manhattan Distance between ', point1, ' and ', point2, ' is: ', manhattan_distance)
