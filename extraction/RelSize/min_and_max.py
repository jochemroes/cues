import numpy as np

eigen_min = np.asarray([
    -54, 0, -305, 0, -380,
    -176, 0, 0, 0, 0,
    -297, 0, 0, 0, -97,
    -145, 0, 0, 0, -34,
    -124, -26, 0, -8, 0,
    0, 0, 0, 0, 0,
    0, -40
])

eigen_max = np.asarray([
    57, 496, 0, 118, 0,
    0, 235, 307, 62, 87,
    0, 218, 129, 101, 0,
    0, 205, 158, 167, 0,
    0, 0, 38, 10, 66,
    31, 27, 121, 74, 103,
    157, 0
])

virt_min = np.asarray(
    [-36,-30,-22,-22,-22,-21,-22,-28,-22,-23,-15,-23,-18,-18,-12,-16,-16,-12,-16,-12,-12,-16,-12,-10,-12,-14,-18,-10,-10,-18,-11,-11]
)

virt_max = np.asarray(
    [74,54,68,36,40,51,42,20,27,35,24,28,19,35,35,16,14,52,22,25,24,18,15,12,18,16,14,11,18,16,15,14]
)
