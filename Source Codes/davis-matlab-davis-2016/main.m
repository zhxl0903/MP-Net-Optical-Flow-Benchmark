% main.m
addpath(fullfile(db_matlab_root_dir, 'db_util'));
addpath(fullfile(db_matlab_root_dir, 'measures'));

% evaluates result for algorithm given algorithm name, tests, and gt: val, train, or all
[eval, raw_eval] = eval_result('LDOF_MP_Net_CRF', {'J', 'F', 'T'}, 'all');
