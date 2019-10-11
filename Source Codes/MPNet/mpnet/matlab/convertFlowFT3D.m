function convertFlowFT3D(set) 
    FT3DPath = '/scratch/clear/ptokmako/datasets/FlyingThings';
    splits = dir([FT3DPath, '/frames_finalpass/', set]);
    for i = 3 : length(splits)
        videos = dir([FT3DPath, '/frames_finalpass/', set, '/', splits(i).name]);
        for j = 3 : length(videos)
            convertSeq2AngleField(FT3DPath, [set, '/', splits(i).name '/', ...
                videos(j).name, '/left/'], 'L');
            convertSeq2AngleField(FT3DPath, [set, '/', splits(i).name '/', ...
                videos(j).name, '/right/'], 'R');
        end
    end        
end

function convertSeq2AngleField(FT3DPath, seqPath, flowSuffix)
    frames = dir([FT3DPath, '/frames_finalpass/', seqPath]);   
    path_split = strsplit(seqPath, '/');
    mkdir([FT3DPath, '/flow_angles/', seqPath]);
    fidAngle = fopen([FT3DPath, '/flow_angles/', seqPath, 'minmax.txt'], 'w');
    
    for i = 3 : length(frames)
        frame_name = frames(i).name;
        split = strsplit(frame_name, '.');        
        [flow, ~] = parsePfm([FT3DPath, '/optical_flow/', path_split{1}, ...
            '/', path_split{2}, '/', path_split{3}, '/into_future/', ...
            path_split{4}, '/OpticalFlowIntoFuture_', split{1}, '_', flowSuffix, '.pfm']);
         
        if isempty(flow)
            return;
        end

        frame = imread([FT3DPath, '/frames_finalpass/', seqPath, frame_name]);
        [h, w, ~] = size(frame);
        flow = flow(1 : h, 1 : w, 1:2);
        
        baseVector = zeros(size(flow, 1), size(flow, 2), 2);
        baseVector(:, :, 1) = 1;
        angleField = acos(dot(flow, baseVector, 3) ./ (sqrt(flow(:, :, 1).^2 ...
            + flow(:, :, 2).^2) .* ones(size(flow, 1), size(flow, 2))));
        magnitudes = sqrt(flow(:, :, 1).^2 + flow(:, :, 2).^2);
        
        minAngle = min(angleField(:));
        maxAngle = max(angleField(:));
        angleField = (angleField - minAngle) ./ (maxAngle - minAngle);
        
        imwrite(angleField, [FT3DPath, '/flow_angles/', seqPath, ...
            'angleField_', split{1}, '.jpg']);
        
        min_magnitude = min(magnitudes(:));
        max_magnitude = max(magnitudes(:));       
        magnitudes = (magnitudes - min_magnitude) ./ (max_magnitude - ...
            min_magnitude);
        
        imwrite(magnitudes, [FT3DPath, '/flow_angles/', seqPath, ...
            'magField_', split{1}, '.jpg']);
        fprintf(fidAngle, '%f %f %f %f\n', minAngle, maxAngle, min_magnitude, ...
            max_magnitude);
    end   
    
    fclose(fidAngle);
end