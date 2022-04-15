#!/usr/bin/env/python3

import random as rand
import math


class Position:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t


class Record:
    def __init__(self, known_positions, position_to_predict):
        self.known_positions = known_positions
        self.position_to_predict = position_to_predict


# Read the positions from a file
positions = [Position(rand.random(), rand.random(), t) for t in range(1000)]

# The number of points to treat as "known"
num_points_per_record = 9

# find the index of the last record
num_positions = len(positions)
max_i = num_positions - num_points_per_record - 1

records = []
for i in range(max_i):
    i_record_end = i + num_points_per_record
    i_position_to_predict = i_record_end + 1
    records.append(
        Record(positions[i:i_record_end],
               positions[i_position_to_predict]))

# Randomize the records so we can split off training and testing sets
rand.shuffle(records)
training_set = records[0:math.ceil(num_positions*0.8)]
testing_set = records[math.ceil(num_positions*0.8):num_positions]
