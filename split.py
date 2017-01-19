import os

data = open('data.txt', 'w')

with open('rewards.txt') as f:
    next(f)
    for line in f:
        #do something


	data.write(line.split()[1] + '\n')


os.system('octave test.m')
