% ------------------------------------------------------------------------
% Jordi Pont-Tuset - http://jponttuset.github.io/
% April 2016
% ------------------------------------------------------------------------
% This file is part of the DAVIS package presented in:
%   Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams,
%   Luc Van Gool, Markus Gross, Alexander Sorkine-Hornung
%   A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
%   CVPR 2016
% Please consider citing the paper if you use this code.
% ------------------------------------------------------------------------
function [db_seq_list, stab_seqs]= db_seqs(subset)

if ~exist('subset','var')
    subset = 'all';
end

if strcmp(subset,'all')
     db_seq_list = {'bear'
                    'blackswan'
                    'bmx-bumps'
                    'bmx-trees'
                    'boat'
                    'breakdance'
                    'breakdance-flare'
                    'bus'
                    'camel'
                    'car-roundabout'
                    'car-shadow'
                    'car-turn'
                    'cows'
                    'dance-jump'
                    'dance-twirl'
                    'dog'
                    'dog-agility'
                    'drift-chicane'
                    'drift-straight'
                    'drift-turn'
                    'elephant'
                    'flamingo'
                    'goat'
                    'hike'
                    'hockey'
                    'horsejump-high'
                    'horsejump-low'
                    'kite-surf'
                    'kite-walk'
                    'libby'
                    'lucia'
                    'mallard-fly'
                    'mallard-water'
                    'motocross-bumps'
                    'motocross-jump'
                    'motorbike'
                    'paragliding'
                    'paragliding-launch'
                    'parkour'
                    'rhino'
                    'rollerblade'
                    'scooter-black'
                    'scooter-gray'
                    'soapbox'
                    'soccerball'
                    'stroller'
                    'surf'
                    'swing'
                    'tennis'
                    'train'};
elseif strcmp(subset,'train')
     db_seq_list = {'bear'
                    'bmx-bumps'
                    'boat'
                    'breakdance-flare'
                    'bus'
                    'car-turn'
                    'dance-jump'
                    'dog-agility'
                    'drift-turn'
                    'elephant'
                    'flamingo'
                    'hike'
                    'hockey'
                    'horsejump-low'
                    'kite-walk'
                    'lucia'
                    'mallard-fly'
                    'mallard-water'
                    'motocross-bumps'
                    'motorbike'
                    'paragliding'
                    'rhino'
                    'rollerblade'
                    'scooter-gray'
                    'soccerball'
                    'stroller'
                    'surf'
                    'swing'
                    'tennis'
                    'train'};
elseif strcmp(subset,'val')
     db_seq_list = {'blackswan'
                    'bmx-trees'
                    'breakdance'
                    'camel'
                    'car-roundabout'
                    'car-shadow'
                    'cows'
                    'dance-twirl'
                    'dog'
                    'drift-chicane'
                    'drift-straight'
                    'goat'
                    'horsejump-high'
                    'kite-surf'
                    'libby'
                    'motocross-jump'
                    'paragliding-launch'
                    'parkour'
                    'scooter-black'
                    'soapbox'};
end


if nargout>1
    % List of sequences where stability is not computed (occlusions, etc.)
    unstab_seqs = {'lucia', 'rollerblade', 'mallard-fly', 'bmx-bumps', 'goat', 'scooter-gray', 'bmx-trees', 'dance-twirl', 'motocross-jump', 'soccerball', 'breakdance', 'dog', 'horsejump-high', 'motorbike', 'breakdance-flare', 'dog-agility', 'horsejump-low', 'paragliding', 'drift-chicane', 'swing','parkour','tennis','libby'};
    
    % Find the stable ones
    stab_seqs = setdiff(1:length(db_seq_list),find(ismember(db_seq_list,unstab_seqs)));
end
  

end
