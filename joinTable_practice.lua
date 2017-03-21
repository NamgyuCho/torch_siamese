require "torch"
require "nn"
require "optim"
require "math"

require "cudnn"
require "cutorch"
require "cunn"

libs = {}

use_cuda = true

if use_cuda == true then
	libs["SpatialConvolution"] = cudnn.SpatialConvolution
	libs["SpatialMaxPooling"] = cudnn.SpatialMaxPooling
	libs["ReLU"] = cudnn.ReLU
	torch.setdefaulttensortype("torch.CudaTensor")
else
	libs["SpatialConvolution"] = nn.SpatialConvolution
	libs["SpatialMaxPooling"] = nn.SpatialMaxPooling
	libs["ReLU"] = nn.ReLU
end

local SpatialConvolution = libs["SpatialConvolution"]
local SpatialMaxPooling = libs["SpatialMaxPooling"]
local ReLU = libs["ReLU"]



-- Encoder / Embedding
-- Input dims are 300x300x3
encoder = nn.Sequential()
encoder:add(SpatialConvolution(3, 24, 3, 3))
encoder:add(ReLU())
encoder:add(SpatialMaxPooling(2, 2, 2, 2))
encoder:add(SpatialConvolution(24, 24, 3, 3))
encoder:add(ReLU())
encoder:add(SpatialMaxPooling(2, 2, 2, 2))
encoder:add(nn.View(24 * 11 * 11)) -- reshapes to view data at 50x34x34

-- The siamese model
siamese_encoder = nn.ParallelTable()
siamese_encoder:add(encoder)
siamese_encoder:add(encoder:clone("weight", "bias", "gradWeight", "gradBias")) -- clone the endoer and share the weight

-- The siamese model (inputs will be Tensors of shape (2, channel, height, width))
model = nn.Sequential()
--model:add(nn.SplitTable(1)) -- split input tensor along the rows (1st dimension) to table for input to ParallelTable
model:add(siamese_encoder)

-- Join two outputs of the siamese network
model:add(nn.JoinTable(1,1)) 

-- Add a set of fully connected (Linear) layers
model:add(nn.Linear(24 * 11 * 11 * 2, 100))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(100, 8))

criterion = nn.MSECriterion()

if use_cuda == true then
	model:cuda()
	criterion:cuda()
	input = torch.randn(torch.LongStorage{2, 3, 50, 50}):cuda()
else
	input = torch.randn(torch.LongStorage{2, 3, 50, 50})
end

print(model)

print("First prediction")

print(#input)
pred = model:forward(input)
print(#pred)

for i = 1, 10 do 	-- a few steps of training such a network..
	if use_cuda == true then
		x = torch.ones(torch.LongStorage{2, 3, 50, 50}):cuda()
		y = torch.ones(8):cuda()
	else
		x = torch.ones(torch.LongStorage{2, 3, 50, 50})
		y = torch.ones(8)
	end

	pred = model:forward(x)

	local err = criterion:forward(pred, y)
	local gradCriterion = criterion:backward(pred, y)

	model:zeroGradParameters()
	model:backward(x, gradCriterion)
	model:updateParameters(0.05)
end
