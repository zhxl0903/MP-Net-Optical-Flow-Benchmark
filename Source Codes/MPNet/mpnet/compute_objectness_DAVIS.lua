--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Run full scene inference in sample image
------------------------------------------------------------------------------]]

require 'torch'
require 'cutorch'
require 'image'

--------------------------------------------------------------------------------
-- parse arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('evaluate deepmask/sharpmask')
cmd:text()
cmd:argument('-model', 'path to model to load')
cmd:text('Options:')
cmd:option('-img','data/testImage.jpg' ,'path/to/test/image')
cmd:option('-gpu', 1, 'gpu device')
cmd:option('-np', 5,'number of proposals to save in test')
cmd:option('-si', -2.5, 'initial scale')
cmd:option('-sf', .5, 'final scale')
cmd:option('-ss', .5, 'scale step')
cmd:option('-dm', false, 'use DeepMask version of SharpMask')

local config = cmd:parse(arg)

--------------------------------------------------------------------------------
-- various initializations
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(config.gpu)

local coco = require 'coco'
local maskApi = coco.MaskApi

local meanstd = {mean = { 0.485, 0.456, 0.406 }, std = { 0.229, 0.224, 0.225 }}

--------------------------------------------------------------------------------
-- load moodel
paths.dofile('DeepMask.lua')
paths.dofile('SharpMask.lua')

print('| loading model file... ' .. config.model)
local m = torch.load(config.model..'/model.t7')
local model = m.model
model:inference(config.np)
model:cuda()

--------------------------------------------------------------------------------
-- create inference module
local scales = {}
for i = config.si,config.sf,config.ss do table.insert(scales,2^i) end

if torch.type(model)=='nn.DeepMask' then
  paths.dofile('InferDeepMask.lua')
elseif torch.type(model)=='nn.SharpMask' then
  paths.dofile('InferSharpMask.lua')
end

local infer = Infer{
  np = config.np,
  scales = scales,
  meanstd = meanstd,
  model = model,
  dm = config.dm,
}

--------------------------------------------------------------------------------
-- do it
print('| start')
local davisPath = '/scratch/clear/ptokmako/datasets/DAVIS'
os.execute("mkdir " .. davisPath .. '/' .. 'Objectness100')
os.execute("mkdir " .. davisPath .. '/' .. 'Objectness100/480p')
local file = io.open(davisPath .. '/ImageSets/480p/trainval.txt')
for line in file:lines() do
	local input, label = line:match("([^ ]+) ([^ ]+)")

	-- load image
	local img = image.load(davisPath .. input)
	local h,w = img:size(2),img:size(3)

	-- forward all scales
	infer:forward(img)

	-- get top propsals
	local masks, scores = infer:getTopProps(.2,h,w)

	local resPath = string.gsub(input, 'JPEGImages', 'Objectness100');
	resPath = string.gsub(resPath, 'jpg', 'png');
	local resultDir = string.gsub(resPath, '%d+.png', '');
	if not path.exists(davisPath .. '/' .. resultDir) then
		os.execute("mkdir " .. davisPath .. '/' .. resultDir)
	end
	
	image.save(davisPath .. '/' .. resPath, masks:sum(1));

end

print('| done')
collectgarbage()
