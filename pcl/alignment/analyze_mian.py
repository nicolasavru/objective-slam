import itertools
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import os.path
import sys

def ReadOcclusionTxt(path):
    data = []
    with open(path) as f:
        for line in f.readlines()[1:]:
            data.append([datum.strip() for datum in line.split()])

    for datum in data:
        datum[2] = float(datum[2])
    return data

def ReadAlignmentOutputDir(data, path):
    for name in os.listdir(path):
        scene_num = name[2:name.index('_')]
        with open(os.path.join(path, name)) as f:
            for line in f:
                if 'Transformations for' in line:
                    cur_model = line.split()[4].split('/')[-1].split('_')[0]
                    if cur_model == 'cheff':
                        cur_model = 'chef'
                    elif cur_model == 'T-rex':
                        cur_model = 'trex'
                    elif cur_model == 'parasaurolophus':
                        cur_model = 'para'
                if 'Distance' in line:
                    distance = line.split()[5:]
                    distance = [float(dist.strip(' ,')) for dist in distance]
                    data_row = [row for row in data if
                                row[0] == scene_num and
                                row[1] == cur_model]
                    print(scene_num, cur_model, data_row)
                    if data_row:
                        data_row[0].append(distance)

MODEL_DIAMS = {
    'chef': 136.59418,
    'trex': 98.828925,
    'para': 131.250275,
    'chicken': 86.28052
}

TWELVEDEG = 0.209440

def MatchWithinThreshold(data, dist_thresh_factor, rot_thresh):
    for d in data:
        d.append([
            d[3][0] <= dist_thresh_factor*MODEL_DIAMS[d[1]],
            d[3][1] <= rot_thresh
        ])

def NormalizeDist(data):
    for d in data:
        d.append([
            d[3][0]/MODEL_DIAMS[d[1]],
        ])

def PercentMatchBelow(data):
    matches = []
    for d in data:
        matches.append(1 if all(d[4]) else 0)
    cum_matches = list(itertools.accumulate(matches, func=operator.add))
    percent_below_matches = []
    for i in range(len(cum_matches)):
        percent_below_matches.append(cum_matches[i]/(i+1))
    return percent_below_matches


data_gpu = ReadOcclusionTxt('UWA/occlusion.txt')
ReadAlignmentOutputDir(data_gpu, '../pcl/alignment/build/run1_gpu')
MatchWithinThreshold(data_gpu, 0.3, TWELVEDEG)
NormalizeDist(data_gpu)
data_gpus = sorted(data_gpu, key=lambda d: d[2])
pd_gpu = PercentMatchBelow(data_gpus)

data_cpu = ReadOcclusionTxt('UWA/occlusion.txt')
ReadAlignmentOutputDir(data_cpu, '../pcl/alignment/build/run1_cpu')
MatchWithinThreshold(data_cpu, 0.3, TWELVEDEG)
NormalizeDist(data_cpu)
data_cpus = sorted(data_cpu, key=lambda d: d[2])
pd_cpu = PercentMatchBelow(data_cpus)


x_gpu = [d[2] for d in data_gpus]
y_gpu = [d[5][0] for d in data_gpus]
r_gpu = [d[3][1] for d in data_gpus]
r_gpu = [d if d <= np.pi else 2*np.pi - d for d in r_gpu]
x_cpu = [d[2] for d in data_cpus]
y_cpu = [d[5][0] for d in data_cpus]
r_cpu = [d[3][1] for d in data_cpus]
r_cpu = [d if d <= np.pi else 2*np.pi - d for d in r_cpu]

x_gpu2 = [d[2] for d in data_gpus if d[2] <= 80 and d[5][0] <= 2]
y_gpu2 = [d[5][0] for d in data_gpus if d[2] <= 80 and d[5][0] <= 2]
r_gpu2 = [d[3][1] for d in data_gpus if d[2] <= 80 and d[5][0] <= 2]
r_gpu2 = [d if d <= np.pi else 2*np.pi - d for d in r_gpu2]
x_cpu2 = [d[2] for d in data_cpus if d[2] <= 80 and d[5][0] <= 2]
y_cpu2 = [d[5][0] for d in data_cpus if d[2] <= 80 and d[5][0] <= 2]
r_cpu2 = [d[3][1] for d in data_cpus if d[2] <= 80 and d[5][0] <= 2]
r_cpu2 = [d if d <= np.pi else 2*np.pi - d for d in r_cpu2]

plt.plot(x_gpu, y_gpu, 'ro', markersize=4, label='GPU')
plt.plot(x_cpu, y_cpu, 'bo', markersize=4, label='CPU')
plt.plot(x_gpu, [np.mean(y_gpu)]*len(x_gpu), 'r-', markersize=4, label='GPU Mean')
plt.plot(x_cpu, [np.mean(y_cpu)]*len(x_cpu), 'b-', markersize=4, label='CPU Mean')
plt.plot(x_gpu, [np.median(y_gpu)]*len(x_gpu), 'r--', markersize=4, label='GPU Median')
plt.plot(x_cpu, [np.median(y_cpu)]*len(x_cpu), 'b--', markersize=4, label='CPU Median')
plt.legend(loc='upper left', fontsize=20)
plt.title('Translation Error', fontsize=30)
plt.xlabel('Percent Occlusion', fontsize=30)
plt.ylabel('Distance in Multiples of Model Diameter', fontsize=30)
plt.tick_params(which='both', labelsize=30)
plt.show()

plt.plot(x_gpu2, y_gpu2, 'ro', markersize=4, label='GPU')
plt.plot(x_cpu2, y_cpu2, 'bo', markersize=4, label='CPU')
plt.plot(x_gpu2, [np.mean(y_gpu2)]*len(x_gpu2), 'r-', markersize=4, label='GPU Mean')
plt.plot(x_cpu2, [np.mean(y_cpu2)]*len(x_cpu2), 'b-', markersize=4, label='CPU Mean')
plt.plot(x_gpu2, [np.median(y_gpu2)]*len(x_gpu2), 'r--', markersize=4, label='GPU Median')
plt.plot(x_cpu2, [np.median(y_cpu2)]*len(x_cpu2), 'b--', markersize=4, label='CPU Median')
plt.legend(loc='upper left', fontsize=20)
plt.title('Translation Error for Occlusion < 80% and with Single Outlier Removed', fontsize=30)
plt.xlabel('Percent Occlusion', fontsize=30)
plt.ylabel('Distance in Multiples of Model Diameter', fontsize=30)
plt.tick_params(which='both', labelsize=30)
plt.show()


# plt.plot(x_gpu, y_gpu, 'ro', x_cpu, y_cpu, 'bo',
#          x_gpu, [np.median(y_gpu)]*len(x_gpu), 'r-',
#          x_cpu, [np.median(y_cpu)]*len(x_cpu), 'b-',
#          markersize=4)
# plt.show()

# plt.plot(x_gpu2, y_gpu2, 'ro', x_cpu2, y_cpu2, 'bo',
#          x_gpu2, [np.median(y_gpu2)]*len(x_gpu2), 'r-',
#          x_cpu2, [np.median(y_cpu2)]*len(x_cpu2), 'b-',
#          markersize=4)
# plt.show()


plt.plot(x_gpu, r_gpu, 'ro', markersize=4, label='GPU')
plt.plot(x_cpu, r_cpu, 'bo', markersize=4, label='CPU')
plt.plot(x_gpu, [np.mean(r_gpu)]*len(x_gpu), 'r-', markersize=4, label='GPU Mean')
plt.plot(x_cpu, [np.mean(r_cpu)]*len(x_cpu), 'b-', markersize=4, label='CPU Mean')
plt.plot(x_gpu, [np.median(r_gpu)]*len(x_gpu), 'r--', markersize=4, label='GPU Median')
plt.plot(x_cpu, [np.median(r_cpu)]*len(x_cpu), 'b--', markersize=4, label='CPU Median')
plt.legend(loc='upper left', fontsize=20)
plt.title('Rotation Error', fontsize=30)
plt.xlabel('Percent Occlusion', fontsize=30)
plt.ylabel('Distance in Radians', fontsize=30)
plt.tick_params(which='both', labelsize=30)
plt.show()

plt.plot(x_gpu2, r_gpu2, 'ro', markersize=4, label='GPU')
plt.plot(x_cpu2, r_cpu2, 'bo', markersize=4, label='CPU')
plt.plot(x_gpu2, [np.mean(r_gpu2)]*len(x_gpu2), 'r-', markersize=4, label='GPU Mean')
plt.plot(x_cpu2, [np.mean(r_cpu2)]*len(x_cpu2), 'b-', markersize=4, label='CPU Mean')
plt.plot(x_gpu2, [np.median(r_gpu2)]*len(x_gpu2), 'r--', markersize=4, label='GPU Median')
plt.plot(x_cpu2, [np.median(r_cpu2)]*len(x_cpu2), 'b--', markersize=4, label='CPU Median')
plt.legend(loc='upper left', fontsize=20)
plt.title('Rotation Error for Occlusion < 80% and with Single Outlier Removed', fontsize=30)
plt.xlabel('Percent Occlusion', fontsize=30)
plt.ylabel('Distance in Radians', fontsize=30)
plt.tick_params(which='both', labelsize=30)
plt.show()


# plt.plot(x_gpu, r_gpu, 'ro', x_cpu, r_cpu, 'bo',
#          x_gpu, [np.median(r_gpu)]*len(x_gpu), 'r-',
#          x_cpu, [np.median(r_cpu)]*len(x_cpu), 'b-',
#          markersize=4)
# plt.show()

# plt.plot(x_gpu2, r_gpu2, 'ro', x_cpu2, r_cpu2, 'bo',
#          x_gpu2, [np.median(r_gpu2)]*len(x_gpu2), 'r-',
#          x_cpu2, [np.median(r_cpu2)]*len(x_cpu2), 'b-',
#          markersize=4)
# plt.show()


print(np.mean(y_gpu), np.median(y_gpu))
print(np.mean(y_cpu), np.median(y_cpu))

print(np.mean(r_gpu), np.median(r_gpu))
print(np.mean(r_cpu), np.median(r_cpu))

# plt.plot(range(len(pd_gpu)), pd_gpu, 'g', range(len(pd_cpu)), pd_cpu, 'b')
# plt.show()
