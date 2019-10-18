% main.m
addpath(fullfile(db_matlab_root_dir, 'db_util'));
addpath(fullfile(db_matlab_root_dir, 'measures'));

[eval, raw_eval] = eval_result('tsp', {'J', 'F', 'T'}, 'all');
