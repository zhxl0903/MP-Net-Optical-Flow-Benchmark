function computeFlowDAVIS()  
    davisPath = '/home/zhang205/Github/Datasets/DAVIS';
    addpath(genpath('.'))
    seqs = dir([davisPath, '/JPEGImages/480p']);
    
    times_list = [];
    for i = 3 : length(seqs)
        seqs(i).name
        times_list = [times_list computeFlowSeq(davisPath, seqs(i).name)];
    end        
    fprintf('Average Time: %.8fs / flowmap\n', mean(times_list));
end

function t = computeFlowSeq(davisPath, seqName)
    frames = dir([davisPath, '/JPEGImages/480p/', seqName]);
    mkdir([davisPath, '/OpticalFlow/480p/', seqName])
    fid = fopen([davisPath, '/OpticalFlow/480p/', seqName, '/minmax.txt'], 'w');
    
    t=[];
    for i = 3 : length(frames) - 1
        frame_name = frames(i).name;
        split = strsplit(frame_name, '.');
        frame1 = imread([davisPath, '/JPEGImages/480p/', seqName, '/' ...
            , frames(i).name]);
        frame2 = imread([davisPath, '/JPEGImages/480p/', seqName, '/' ...
            , frames(i + 1).name]);
        
        [flow, ~] = sundaramECCV10_ldof_GPU_mex(frame1, frame2);
        
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
        
        time_elapsed = toc;
        t = [t time_elapsed];
        
        imwrite(magnitudes, [davisPath, '/OpticalFlow/480p/', seqName ...
            , '/magField_', split{1}, '.jpg']);
        
        fprintf(fid, '%f %f %f %f\n', minAngle, maxAngle, minMagnitude, maxMagnitude);
    end

    fclose(fid);  
    return;
end
