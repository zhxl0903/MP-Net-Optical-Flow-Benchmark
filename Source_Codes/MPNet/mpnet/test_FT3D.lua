require 'lfs'
require 'cutorch'
require 'image'
require 'segment_FT3D_angle'
require 'ResizeJoinTable'

function testVideo(modle, path)
    local preds = segment(modle, path)

    if not preds then
        return preds
    end

    local frameNames = {}
    for file in lfs.dir(path) do
        if lfs.attributes(file, "mode") ~= "directory" and file ~= 'motion_prediction' then
            table.insert(frameNames, file)
        end
    end

    table.sort(frameNames)

    local iou = 0
    for i = 1, #frameNames do
        local gt = image.load(path .. '/' .. frameNames[i])
        iou = iou + computeFrameIOU(gt, torch.round(preds[i]))
    end

    return iou / #frameNames
end

function computeFrameIOU(gt, predicted)
    local intersection = torch.cmul(gt, predicted):sum()
    local union = torch.add(gt, predicted):gt(0):sum()

    if union == 0 then
        return 1;
    end

    return intersection / union;
end

cmd = torch.CmdLine()
cmd:option('-gpu', 0, 'GPU id')
cmd:option('-model', '', 'model to test')

local params = cmd:parse(arg)

cutorch.setDevice(params.gpu + 1)
local model = torch.load('models/' .. params.model):float()
model = model:cuda()
model:evaluate()

local iou = 0;
local count = 0;
local FT3Dpath = '/scratch/clear/ptokmako/datasets/FlyingThings';
local file = io.open(FT3Dpath .. '/testList.txt')
for line in file:lines() do
    local iou_vid = testVideo(model, FT3Dpath .. '/motion_labels/' .. line)
    if iou_vid then
        iou = iou + iou_vid;
        count = count + 1;
        print(line .. ': ' .. iou_vid)
    end
    collectgarbage()
end

print(iou / count)