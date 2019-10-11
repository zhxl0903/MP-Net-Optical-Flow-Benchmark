require 'image'
require 'lfs'
require 'cunn'
require 'nngraph'

function segment(model, rgbPath, setting)
    local flowPath = string.gsub(rgbPath, 'JPEGImages', 'OpticalFlow');

    local fileName = string.gsub(flowPath, '%d+.jpg', 'minmax.txt')
    local file = io.open(fileName)
    local minmaxes = {}
    local ind = 1;
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

    local resized_width = 424
    local resized_height = 232
    local batch = torch.Tensor(1, 2, resized_height, resized_width);

    local frameNum = tonumber(flowPath:match('(%d+).jpg')) + 1
    local mm = minmaxes[frameNum]
    local angleFileName = string.gsub(flowPath, '(%d+).jpg', 'angleField_%1.jpg');
    local magFileName = string.gsub(flowPath, '(%d+).jpg', 'magField_%1.jpg');

    local f = io.open(angleFileName, "r")
    if not f then
        return nil
    end
    io.close(f)

    local flowAngle = image.load(angleFileName)
    local flowMag = image.load(magFileName)

    local flowFrame = torch.cat(flowAngle, flowMag, 1)
    flowFrame = image.scale(flowFrame, resized_width, resized_height, 'simple');

    flowFrame[{{1}, {}, {}}] = flowFrame[{{1}, {}, {}}] * (mm[2] - mm[1]) + mm[1]
    flowFrame[{{2}, {}, {}}] = flowFrame[{{2}, {}, {}}] * (mm[4] - mm[3]) + mm[3]

    flowFrame[{{1}, {}, {}}]:cmul(flowFrame[{{2}, {}, {}}]:gt(1):double())

    flowFrame[{{2}, {}, {}}]:div(math.sqrt(math.pow(854, 2) + math.pow(480, 2)) / 6)

    batch[1] = flowFrame;

    batch = batch:float():cuda()

    local outputs = model:forward(batch)

    local objPath = string.gsub(rgbPath, 'JPEGImages', 'Objectness100');
    objPath = string.gsub(objPath, '(%d+).jpg', '%1.png');
    local objectness = image.load(objPath)

    local preds
    preds = torch.Tensor(1, 480, 854)
    local pred = outputs[1];
    pred = nn.utils.recursiveType(pred, 'torch.DoubleTensor')
    pred = image.scale(pred, 854, 480)

    local predRaw = pred
    objectness:div(objectness:max()):add(0.5)
    pred:cmul(objectness)
    pred:clamp(0, 1)

    preds[1] = pred

    local resultPath = string.gsub(rgbPath, 'JPEGImages', 'Results/Segmentations');
    local resultPath = string.gsub(resultPath, '480p', '480p/' .. setting);
    local resultPath = string.gsub(resultPath, 'jpg', 'png');
    local resultDir = string.gsub(resultPath, '%d+.png', '');
    if not path.exists(resultDir) then
        os.execute("mkdir " .. resultDir)
    end
    image.save(resultPath, torch.round(preds));
    local resultPathRaw = string.gsub(resultPath, '(%d+).png', 'raw_%1.png');
    image.save(resultPathRaw, predRaw);
end
