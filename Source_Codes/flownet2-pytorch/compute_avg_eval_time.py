txt_file_path = 'time_benchmark.txt'

with open(txt_file_path, 'r') as f:
    lines = f.readlines()

# Converts lines to list of tuples (n_frame_pairs in seq, eval_time of seq)
lines = list(map(lambda str: (float(str.split(' ')[0]), float(str.split(' ')[1].rstrip('\n'))), lines))

n_frame_pairs = 0.0
total_time = 0.0
# Computes average eval time / frame pair
for (n, t) in lines:
    n_frame_pairs += n
    total_time += t

print('Average eval time: {}s per frame pair'.format(total_time / n_frame_pairs))
