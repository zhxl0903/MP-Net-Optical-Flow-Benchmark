require 'image'
require 'lfs'
require 'cunn'
require 'nngraph'

function segment(model, path)
    local frameNames = {}
    for file in lfs.dir(path) do
        if lfs.attributes(file, "mode") ~= "directory" then
            table.insert(frameNames, file)
        end
    end

    if #frameNames < 10 then
        print('Not enough frames')
        return nil
    end

    table.sort(frameNames)

    local flowPath = string.gsub(path, 'motion_labels', 'flow_angles_no_artifacts');

    local file = io.open(flowPath .. '/' .. 'minmax.txt')
    local minmaxes = {}
    local ind = 1
    if file then
        for line in file:lines() do
            local min_x, max_x, min_y, max_y = unpack(line:split(" "))
            minmaxes[ind]  = {min_x, max_x, min_y, max_y}
            ind = ind + 1
        end
    else
        print('File not found!!!!!!!!')
    end
    io.close(file)

    local resized_width = 520
    local resized_height = 296
    local batch = torch.Tensor(#frameNames, 2, resized_height, resized_width);
    for i = 1, #frameNames do
        local mm = minmaxes[i]
        local angleFileName = 'angleField_' ..  string.gsub(frameNames[i], 'png', 'jpg');
        local magFileName = 'magField_' ..  string.gsub(frameNames[i], 'png', 'jpg');

        local f = io.open(flowPath .. angleFileName, "r")
        if not f then
            print(flowPath .. angleFileName)
            return nil
        end
        io.close(f)

        local flowAngle = image.load(flowPath .. angleFileName)
        local flowMag = image.load(flowPath .. magFileName)

        local flowFrame = torch.cat(flowAngle, flowMag, 1)
        flowFrame = image.scale(flowFrame, resized_width, resized_height, 'simple');

        flowFrame[{{1}, {}, {}}] = flowFrame[{{1}, {}, {}}] * (mm[2] - mm[1]) + mm[1]
        flowFrame[{{2}, {}, {}}] = flowFrame[{{2}, {}, {}}] * (mm[4] - mm[3]) + mm[3]

        flowFrame[{{2}, {}, {}}]:div(math.sqrt(math.pow(960, 2) + math.pow(540, 2)) / 6)

        batch[i] = flowFrame;
    end

    batch = batch:float():cuda()

    local outputs = model:forward(batch)

    local preds = torch.Tensor(#frameNames, 540, 960)
    for i = 1, #frameNames do
        local pred = outputs[i];
        pred = nn.utils.recursiveType(pred, 'torch.DoubleTensor')
        preds[i] = image.scale(pred, 960, 540);
    end

    return preds

end
