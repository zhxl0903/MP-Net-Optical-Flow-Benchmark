davis_path='/home/zhang205/Github/Datasets/DAVIS';

frames_path = fullfile(davis_path, 'JPEGImages', '480p');
flow_path = fullfile(davis_path, 'OpticalFlow', '480p');

% get the folder contents
seqs=dir(frames_path);

% remove all files (isdir property is 0)
seqs=seqs([seqs(:).isdir]==1);

% remove '.' and '..'
seqs=seqs(~ismember({seqs(:).name},{'.','..'}));

for k = 1:length(seqs)
    seq_path = fullfile(seqs(k).folder, seqs(k).name);
    save_path = fullfile(flow_path, seqs(k).name);
    mkdir(save_path);
    
    dir_get_epicflow(seq_path, save_path, 8);
end