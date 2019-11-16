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
% This file defines the parameters of the experiments
% ------------------------------------------------------------------------

% List of techniques compared
techniques = {'LDOF_MP_Net_CRF', 'Epicflow_MP_Net_CRF', 'Flownet2_MP_Net_CRF' };
          
% Names to be shown on the tables
techniques_paper = {'LDOF MP-Net', 'Epicflow_MP_Net_CRF', 'Flownet2_MP_Net_CRF'};
% Output folder to save files
paper_data = './';       
