import numpy as np


# Remove outliers based on Z-axis threshold
def removeOutliers(data, threshold=0.6):

    points = data["points"]
    zMin, zMax = np.min(points[:, 2]), np.max(points[:, 2])
    zRange = zMax - zMin
    zUpperBound = zMin + (threshold * zRange)

    # Mask for points below the upper bound
    mask = np.where(points[:, 2] <= zUpperBound)[0]  # [0] to just get indicies
    for k, v in data.items():
        if k is not None and v is not None:
            data[k] = data[k][mask]

    return data


file_path = "../test_data/test104.npy"
data = np.load(file_path, allow_pickle=True).item()
cleaned_data = removeOutliers(data.copy())
np.save("../test_data_cleaned/test104_cleaned.npy", cleaned_data)


# print(data["points"].shape)
# print(cleaned_data["points"].shape)
# print(data["points"][0:10])
# print(cleaned_data["points"][0:10])
