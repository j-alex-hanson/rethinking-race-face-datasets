import json
import numpy as np

def scale_5k_and_round(point):
    temp_point = (5000 * np.array(point) / 100).tolist()
    print(temp_point)
    remainder_point = []
    for i, entry in enumerate(temp_point):
        temp_point[i] = entry // 1
        remainder = entry - temp_point[i]
        remainder_point.append(remainder)
    remainder_point = np.array(remainder_point)
    while sum(temp_point) < 5000:
        add_idx = np.argmax(remainder_point)
        remainder_point[add_idx] = 0.0
        temp_point[add_idx] += 1
    print(temp_point)
    return temp_point


def list_of_points(high):

    low = (100 - high) / 3.0

    p25 = (high - low) / 4.0 + low
    p50 = (high - low) / 2.0 + low
    p75 = high - (high - low) / 4.0

    points = []

    p = [low] * 4
    p[-1] = high
    p = scale_5k_and_round(p)
    points.append(p)

    for i in range(3):
        p = [low] * 4
        p[i] = p25
        p[-1] = p75
        p = scale_5k_and_round(p)
        points.append(p)

        p = [low] * 4
        p[i] = p50
        p[-1] = p50
        p = scale_5k_and_round(p)
        points.append(p)

        p = [low] * 4
        p[i] = p75
        p[-1] = p25
        p = scale_5k_and_round(p)
        points.append(p)

        p = [low] * 4
        p[i] = high
        p = scale_5k_and_round(p)
        points.append(p)

        p = [low] * 4
        p[i] = p75
        p[(i+1)%3] = p25
        p = scale_5k_and_round(p)
        points.append(p)

        p = [low] * 4
        p[i] = p50
        p[(i+1)%3] = p50
        p = scale_5k_and_round(p)
        points.append(p)

        p = [low] * 4
        p[i] = p25
        p[(i+1)%3] = p75
        p = scale_5k_and_round(p)
        points.append(p)

    return points

percents = (100, 60, 40, 30)

config = {'order': ['African', 'Asian', 'Caucasian', 'Indian'],
          'distributions': {},
          }

for percent in percents:
    config['distributions'][percent] = list_of_points(percent)

with open('points_config.json', 'w') as f:
    json.dump(config, f, indent=2, sort_keys=True)
 
