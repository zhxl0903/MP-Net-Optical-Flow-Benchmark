function computeFlowDAVIS_other_flows()  
    %{
      This function converts flowmaps from flow_method to angle
      and speed maps.
    %}
    davisPath = '/home/zhang205/Github/Datasets/DAVIS';
    flow_method = 'Spynet';
    
    addpath(genpath('.'))
    seqs = dir([davisPath, '/JPEGImages/480p']);
    
    % defines flowmap conversion time
    flow_conversion_times_list = [];

    for i = 3 : length(seqs)
        seqs(i).name
        [t1] = computeFlowSeq(davisPath, flow_method, seqs(i).name);
        flow_conversion_times_list = [flow_conversion_times_list t1];
    end        
    fprintf('Average Flowmap Conversion Time: %.8fs / flowmap\n', mean(flow_conversion_times_list));
end

function [flow_conversion_time] = computeFlowSeq(davisPath, flowMethod, seqName)

    % Gets flow files for seq given flowMethod
    seq_path = fullfile(davisPath, 'OpticalFlow', strcat('480p_', flowMethod), seqName);
    flows = dir([seq_path]);
    
    % Creates save dir for seq seqName
    mkdir([davisPath, '/OpticalFlow/480p/', seqName])
    
    % Creates file to save minmax values
    fid = fopen([davisPath, '/OpticalFlow/480p/', seqName, '/minmax.txt'], 'w');
    
    % Defines flowmap conversion time
    flow_conversion_time=[];
    
    % Loops over each flow file in seq in numeric order
    for i = 3 : length(flows)
        
        % Gets name of flow file
        flow_name = flows(i).name;
        split = strsplit(flow_name, '.');
        
        % Reads flowmap
        flow = readFlowFile(fullfile(flows(i).folder, flow_name));
        
        % Times flowmap conversion time
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
        
        imwrite(magnitudes, [davisPath, '/OpticalFlow/480p/', seqName ...
            , '/magField_', split{1}, '.jpg']);
        
        flow_conversion_time = [flow_conversion_time toc];
        
        fprintf(fid, '%f %f %f %f\n', minAngle, maxAngle, minMagnitude, maxMagnitude);
    end

    fclose(fid);  
    return;
end
