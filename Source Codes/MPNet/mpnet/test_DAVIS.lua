require 'lfs'
require 'segment_DAVIS_angle'
require 'cutorch'
require 'image'
require 'ResizeJoinTable'

cmd = torch.CmdLine()
cmd:option('-gpu', 0, 'GPU id')
cmd:option('-model', '', 'model to test')
cmd:option('-setting', '', 'version of the model')

local params = cmd:parse(arg)
local davisDir = '/scratch/clear/ptokmako/datasets/DAVIS'

os.execute("mkdir " .. davisDir .. "/Results/Segmentations/480p/" .. params.setting)

cutorch.setDevice(params.gpu + 1)
local model = torch.load('models/' .. params.model):float()
model = model:cuda()
model:evaluate()

local file = io.open(davisDir .. '/ImageSets/480p/trainval.txt')
for line in file:lines() do
    local input, label = line:match("([^ ]+) ([^ ]+)")
    segment(model, davisDir .. input, params.setting)
end

print('Finished processing')
