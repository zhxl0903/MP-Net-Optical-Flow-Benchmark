davis_path='/home/zhang205/Github/Datasets/DAVIS';

frames_path = fullfile(davis_path, 'JPEGImages', '480p');
flow_path = fullfile(davis_path, 'OpticalFlow', '480p_Epicflow');

mkdir(flow_path);

% get the folder contents
seqs=dir(frames_path);

% remove all files (isdir property is 0)
seqs=seqs([seqs(:).isdir]==1);

% remove '.' and '..'
seqs=seqs(~ismember({seqs(:).name},{'.','..'}));
flow_times = [];

flow_map_count=0;

for k = 1:length(seqs)
    seq_path = fullfile(seqs(k).folder, seqs(k).name);
    save_path = fullfile(flow_path, seqs(k).name);
    
    % Counts number of flowmaps in seq and increases flow_map_count
    n_flow_maps = length(dir(seq_path))-3;
    flow_map_count = flow_map_count + n_flow_maps;
    
    mkdir(save_path);
    
    tic;
    dir_get_epicflow(seq_path, save_path, 0);
    flow_times = [flow_times, toc];
end

fprintf('Average Flow Estimation Time: %.8fs\n / frame pair', sum(flow_times)/flow_map_count);