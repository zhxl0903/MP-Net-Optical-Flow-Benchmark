require 'lfs'
require 'segment_DAVIS_angle'
require 'cutorch'
require 'image'
require 'ResizeJoinTable'

cmd = torch.CmdLine()
cmd:option('-gpu', 0, 'GPU id')
cmd:option('-model', 'model_paper.dat', 'model to test')
cmd:option('-setting', 'Epicflow_MP_Net', 'version of the model')

local params = cmd:parse(arg)
local davisDir = '/home/zhang205/Github/Datasets/DAVIS'

os.execute("mkdir " .. davisDir .. "/Results/Segmentations/480p/" .. params.setting)

cutorch.setDevice(params.gpu + 1)
local model = torch.load('models/' .. params.model):float()
model = model:cuda()
model:evaluate()

local count = 0
local s = 0

local file = io.open(davisDir .. '/ImageSets/480p/trainval.txt')
for line in file:lines() do
    local input, label = line:match("([^ ]+) ([^ ]+)")
    
    --gets directory of flowmap
    local flowPath = string.gsub(davisDir .. input, 'JPEGImages', 'OpticalFlow')

    print(string.format("Processing flowmap: %s\n", flowPath))  

    --defines start time
    local start_time = os.clock()

    segment(model, davisDir .. input, params.setting)
    
    --defines elapsed time
    local elapsed_time = os.clock() - start_time
    
    print(string.format("Elapsed Time: %.8fs\n", elapsed_time)) 
    
    count = count + 1
    s = s + elapsed_time
end

print(string.format("Average Time: %.8fs / flowmap\n", s / count))
print('Finished processing')
