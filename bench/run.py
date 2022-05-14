import subprocess
import threading
import numpy as np
import pandas as pd
import os
import sys

RESOLUTION = 64
N = 1000
N_threads = 6
FILE_PATH = sys.path[0]
RADIATION_LENGTHS = [30423.2, 47.9, 42.4, 36.1, 34.4, 15.8, 10.7,  8.9, 1.76, 0.56] # air polyethene polystyrene water plexiglass teflon concrete aluminum iron lead 

def run_threads(run, i, voxels):
    bins = np.arange(-12, 12 + 12/RESOLUTION,24/RESOLUTION)
    np.savetxt("run_" + str(i) + ".voxel", voxels.flatten(), fmt="%i", delimiter=" ")
    args = ["mu", "run.mac", "run_" + str(i)]
    proc = subprocess.Popen(args = [os.path.join(FILE_PATH, "mu"), "run.mac", "run_" + str(i)], stdout=subprocess.DEVNULL)
    proc.wait()
    print(i)
    txt_df = pd.read_csv("run_" + str(i) + ".txt", delimiter=" ", header=None, 
                                 names=["event", "count", "x", "y", "z", "time", "eIn", "eDep", 
                                          "trackID", "copyNo", "particleID"])
    txt_df = txt_df[(txt_df["particleID"] == 13) & (txt_df["x"] > 150)][["y", "z", "count"]]
    txt_df["y_cut"] = pd.cut(txt_df["y"], bins = bins, right = False)
    txt_df["z_cut"] = pd.cut(txt_df["z"], bins = bins, right = False)
    pt = pd.pivot_table(txt_df, columns = "y_cut", index = "z_cut", values="count", aggfunc="sum")
    np.save("detections/" + str(run) + "_orient_" + str(i) + ".npy", pt.values)


def rotate_cube(cuberay):
    res = []
    res.append(cuberay)
    res.append(np.rot90(cuberay, 2, axes=(0,2)))
    res.append(np.rot90(cuberay, axes=(0,2)))
    res.append(np.rot90(cuberay, -1, axes=(0,2)))
    res.append(np.rot90(cuberay, axes=(0,1)))
    res.append(np.rot90(cuberay, -1, axes=(0,1)))
    return res

def main():
    threads = []
    for j in range(N):
        voxels = np.zeros((64, 64, 64), dtype="int")
        radations_lengths = np.full((64, 64, 64), RADIATION_LENGTHS[0])
        for i in range(int(np.round(np.random.uniform(5, 1)))):
            material = np.random.randint(1, 10) 
            voxels = generate_blob(voxels, material)
            radations_lengths[voxels == material] = RADIATION_LENGTHS[material]
        orientations = rotate_cube(voxels)
        for i in range(N_threads):
            th = threading.Thread(target=run_threads, args=(j, i, orientations[i]))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        np.save("voxels/run_" + str(j) + ".npy", radations_lengths)
        print("Run", j)

# generate blob using metaballs
def generate_blob(arr, fill_value=1, center=None, r=0.1, std_dev=0.05, n_centroids=None):
    # center of blob
    if center is None:
        center = np.random.random(3)

    r_2 = np.random.normal(r, std_dev) ** 2
    xx, yy, zz = np.mgrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]] / arr.shape[0]
    arr[(xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2 < r_2] = fill_value

    # number of metaballs
    if n_centroids is None:
        n_centroids = np.round(np.random.normal(20, 3)).astype("int")
    if n_centroids < 1: n_centroids = 0

    for i in range(n_centroids):
        # center of metaball
        d = np.random.normal(r, std_dev) # it doesn't actually matter if it goes negative
        metaball_rad = np.random.normal(r, std_dev)
        angle = np.random.uniform(0, 2 * np.pi)
        angle_2 = np.random.uniform(0, 2 * np.pi)
        centroid = center + np.array((metaball_rad * np.sin(angle) * np.cos(angle_2), metaball_rad * np.sin(angle) * np.sin(angle_2), metaball_rad * np.cos(angle)))

        # sphere around centroid
        r_2 = np.random.normal(r, std_dev) ** 2
        xx, yy, zz = np.mgrid[:arr.shape[0], :arr.shape[1], :arr.shape[2]] / arr.shape[0]
        arr[(xx - centroid[0]) ** 2 + (yy - centroid[1]) ** 2 + (zz - centroid[2]) ** 2 < r_2] = fill_value
    return arr

# def fade(t):
#     return t * t * t * (t * (t * 6 - 15) + 10)
    
# def lerp(t, a, b):
#     return a + t * (b - a)

# def grad(ahash, x, y, z):
#     # very screwed up bitwise operations
#     h = ahash & 15
#     u = x if h < 8 else y
#     v = y if h < 4 else x if (h == 12 or h == 14) else z
#     return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)
# 
#     
# class PerlinNoise: # because the library has "undesired behaviour"

#     def __init__(self, seed=None):
#         if seed is None:
#             p = np.arange(256)
#             np.random.shuffle(p)
#             self.p = np.tile(p, 2)
#         else:
#             np.random.seed(seed)
#             p = np.arange(256)
#             np.random.shuffle(p)
#             self.p = np.tile(p, 2)
    
#     def noise(self, x, y, z): # i have no idea what this does
#         X = int(np.floor(x)) & 255
#         Y = int(np.floor(y)) & 255
#         Z = int(np.floor(z)) & 255

#         x -= np.floor(x)
#         y -= np.floor(y)
#         z -= np.floor(z)
        
#         u = fade(x)
#         v = fade(y)
#         w = fade(z)

#         A = self.p[X] + Y
#         AA = self.p[A] + Z
#         AB = self.p[A + 1] + Z
#         B = self.p[X + 1] + Y
#         BA = self.p[B] + Z
#         BB = self.p[B + 1] + Z

#         # ???
#         res = lerp(w, lerp(v, lerp(u, grad(self.p[AA], x, y, z), grad(self.p[BA], x-1, y, z)), lerp(u, grad(self.p[AB], x, y-1, z), grad(self.p[BB], x-1, y-1, z))),	lerp(v, lerp(u, grad(self.p[AA+1], x, y, z-1), grad(self.p[BA+1], x-1, y, z-1)), lerp(u, grad(self.p[AB+1], x, y-1, z-1),	grad(self.p[BB+1], x-1, y-1, z-1))))
#         return (res + 1.0)/2.0


if __name__ == "__main__":
    main()