function computeFlowDAVIS()  
    davisPath = '/home/zhang205/Github/Datasets/DAVIS';
    addpath(genpath('.'))
    seqs = dir([davisPath, '/JPEGImages/480p']);
    
    % defines flowmap conversion time
    flow_conversion_times_list = [];

    % defines flow estimation time
    flow_estimation_time_list = [];
    for i = 3 : length(seqs)
        seqs(i).name
        [t1, t2] = computeFlowSeq(davisPath, seqs(i).name);
        flow_conversion_times_list = [flow_conversion_times_list t1];
        flow_estimation_time_list = [flow_estimation_time_list t2];
    end        
    fprintf('Average Flowmap Conversion Time: %.8fs / flowmap\n', mean(flow_conversion_times_list));
    fprintf('Average LDOF Flow Estimation Time: %.8fs / flowmap\n', mean(flow_estimation_time_list));
end

function [flow_conversion_time, flow_estimation_time] = computeFlowSeq(davisPath, seqName)
    frames = dir([davisPath, '/JPEGImages/480p/', seqName]);
    mkdir([davisPath, '/OpticalFlow/480p/', seqName])
    fid = fopen([davisPath, '/OpticalFlow/480p/', seqName, '/minmax.txt'], 'w');
    
    % Defines flowmap conversion time
    flow_conversion_time=[];
    
    % Defines optical flow estimation time
    flow_estimation_time=[];
    for i = 3 : length(frames) - 1
        frame_name = frames(i).name;
        split = strsplit(frame_name, '.');
        frame1 = imread([davisPath, '/JPEGImages/480p/', seqName, '/' ...
            , frames(i).name]);
        frame2 = imread([davisPath, '/JPEGImages/480p/', seqName, '/' ...
            , frames(i + 1).name]);
        
        tic;
        [flow, ~] = sundaramECCV10_ldof_GPU_mex(frame1, frame2);
        flow_estimation_time = [flow_estimation_time toc];

        tic;
        temp = flow(:, :, 2);
        flow(:, :, 2) = flow(:, :, 1);        
        flow(:, :, 1) = temp;
                
        baseVector = zeros(size(flow, 1), size(flow, 2), 2);        
        baseVector(:, :, 1) = 1;
        
        angleField = acos(dot(flow, baseVector, 3) ./ ...
            ((sqrt(flow(:, :, 1).^2 + flow(:, :, 2).^2)) .* ...
            ones(size(flow, 1), size(flow, 2))));
        magnitudes = sqrt(flow(:, :, 1).^2 + flow(:, :, 2).^2);
        
        minAngle = min(angleField(:));
        maxAngle = max(angleField(:));
        angleField = (angleField - minAngle) ./ (maxAngle - minAngle);
        
        imwrite(angleField, [davisPath, '/OpticalFlow/480p/', seqName ...
            , '/angleField_', split{1}, '.jpg']);
        
        minMagnitude = min(magnitudes(:));
        maxMagnitude = max(magnitudes(:));
        magnitudes = (magnitudes - minMagnitude) ./ (maxMagnitude - minMagnitude);
        flow_conversion_time = [flow_conversion_time toc];
        
        imwrite(magnitudes, [davisPath, '/OpticalFlow/480p/', seqName ...
            , '/magField_', split{1}, '.jpg']);
        
        fprintf(fid, '%f %f %f %f\n', minAngle, maxAngle, minMagnitude, maxMagnitude);
    end

    fclose(fid);  
    return;
end
