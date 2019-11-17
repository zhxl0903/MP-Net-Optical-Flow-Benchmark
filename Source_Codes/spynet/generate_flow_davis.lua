--loads and initializes spynet
spynet = require('spynet')
flowX = require 'flowExtensions'
easyComputeFlow = spynet.easy_setup()

local davisPath = '/home/zhang205/Github/Datasets/DAVIS'

os.execute("mkdir " .. davisPath .. '/' .. 'OpticalFlow/480p_Spynet')
local file = io.open('sequences_path.txt')

local count = 0
local s = 0

--loops over frame paths of DAVIS
for line in file:lines() do

        --Gets paths of input frames
	local input1, input2 = line:match("([^ ]+) ([^ ]+)")
        
	--Gets result dir
        local resultDir = string.gsub(input1, 'JPEGImages', 'OpticalFlow') 
        local resultDir = string.gsub(resultDir, '480p', '480p_Spynet') 
        local resultFlowPath = string.gsub(resultDir, '.jpg', '.flo') 
	local resultDir = string.gsub(resultDir, '%d+.jpg', '');
        
        -- Creates result dir if dir does not exist
        if not path.exists(resultDir) then
           os.execute("mkdir " .. resultDir)
    	end
        
        -- Evaluates Model using input frame pairs
        im1 = image.load(input1)
	im2 = image.load(input2)
        
        -- Benchmarks evaluation time
        print(string.format("Processing flowmap: %s\n", resultFlowPath)) 
        local start_time = os.clock()
	flow = easyComputeFlow(im1, im2)
        local elapsed_time = os.clock() - start_time 
        print(string.format("Elapsed Time: %.8fs\n", elapsed_time))   
	
        flowX.writeFLO(resultFlowPath, flow)
        
        count = count + 1.0
        s = s + elapsed_time
end

print(string.format("Average Time: %.8fs / flowmap\n", s / count))
print('| done')
collectgarbage()
        
        
        







