-- Copyright 2016 Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.  
-- By using this software you agree to the terms of the license file 
-- in the root folder.
-- For commercial use, please contact ps-license@tue.mpg.de.

require 'image'
require 'cutorch'

local cmd = torch.CmdLine()
cmd:option('-data', '../FlyingChairs/data', 'Flying Chairs data directory')
opt = cmd:parse(arg or {})

opt.showFlow = 0
opt.fineHeight = 384
opt.fineWidth = 512
opt.preprocess = 0
opt.level = 5 
opt.polluteFlow = 0
opt.augment = 0
opt.warp = 1
opt.batchSize = 1
local donkey = require('timing_util')

local train_samples, validation_samples = donkey.getTrainValidationSplits('train_val_split.txt')
local loss = torch.zeros(1,1, opt.fineHeight, opt.fineWidth):float()
local errors = torch.zeros(validation_samples:size()[1])
timings = torch.zeros(validation_samples:size()[1])
local loss = 0
local flowCPU = cutorch.createCudaHostTensor(640, 2,opt.fineHeight,opt.fineWidth):uniform()

for i=1,validation_samples:size()[1] do
    collectgarbage()

    local id = validation_samples[i][1]
    local imgs, flow = donkey.testHook(id)

    timer = torch.Timer()
    imgs = imgs:resize(1,6,opt.fineHeight, opt.fineWidth):cuda()
    flow_est = donkey.computeInitFlowL5(imgs):squeeze()
    flowCPU[i]:copyAsync(flow_est)
    cutorch.streamSynchronize(cutorch.getStream())
    local time_elapsed = timer:time().real  

    print('Time Elapsed: '..time_elapsed)

    timings[i] = time_elapsed
end
cutorch.streamSynchronize(cutorch.getStream())


for i=1,validation_samples:size()[1] do
    local id = validation_samples[i][1]
    local raw_im1, raw_im2, raw_flow = donkey.getRawData(id)
    
    local _err = (raw_flow - flowCPU[i]):pow(2)
    local err = torch.sum(_err, 1):sqrt()
    loss = loss + err:float()
    errors[i] = err:mean() 
    
    print(i, errors[i])
end
loss = torch.div(loss, validation_samples:size()[1])
print('Average EPE = '..loss:sum()/(opt.fineWidth*opt.fineHeight))
print('Mean Timing: ' ..timings:mean())
print('Median Timing: ' ..timings:median()[1])
