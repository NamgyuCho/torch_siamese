-- Build a siamese network
-- see https://github.com/torch/nn/issues/757 what they considered 
require "nn"

function build_model(libs)
	local SpatialConvolution = libs["SpatialConvolution"]
	local SpatialMaxPooling = libs["SpatialMaxPooling"]
	local ReLU = libs["ReLU"]

	-- Encoder / Embedding
	-- Input dims are 300x300x3
	encoder = nn.Sequential()
	encoder:add(SpatialConvolution(3, 64, 3, 3))
	encoder:add(ReLU())
	encoder:add(SpatialConvolution(64, 64, 3, 3))
	encoder:add(ReLU())
	encoder:add(SpatialMaxPooling(2, 2, 2, 2))

	encoder:add(SpatialConvolution(64, 128, 3, 3)) 
	encoder:add(ReLU())
	encoder:add(SpatialConvolution(128, 128, 3, 3)) 
	encoder:add(ReLU())
	encoder:add(SpatialMaxPooling(2, 2, 2, 2))

	encoder:add(SpatialConvolution(128, 128, 3, 3)) 
	encoder:add(ReLU())
	encoder:add(SpatialConvolution(128, 128, 3, 3)) 
	encoder:add(ReLU())
	encoder:add(SpatialMaxPooling(2, 2, 2, 2))

	encoder:add(SpatialConvolution(128, 256, 3, 3)) 
	encoder:add(ReLU())
	encoder:add(SpatialConvolution(256, 256, 3, 3)) 
	encoder:add(ReLU())
	encoder:add(SpatialMaxPooling(2, 2, 2, 2))

	finalFilterSize = 256
	finalOutSize = 15

	encoder:add(nn.View(finalFilterSize * finalOutSize * finalOutSize)) -- reshapes to view data at 50x34x34

	-- The siamese model
	siamese_encoder = nn.ParallelTable()
	siamese_encoder:add(encoder)
	siamese_encoder:add(encoder:clone("weight", "bias", "gradWeight", "gradBias")) -- clone the endoer and share the weight

	-- The siamese model (inputs will be Tensors of shape (2, channel, height, width))
	model = nn.Sequential()
	model:add(nn.SplitTable(1)) -- split input tensor along the rows (1st dimension) to table for input to ParallelTable
	model:add(siamese_encoder)

	-- Join two outputs of the siamese network
	model:add(nn.JoinTable(1,1)) 

	-- Add a set of fully connected (Linear) layers
	fcSize = 2048
	model:add(nn.Linear(finalFilterSize * finalOutSize * finalOutSize * 2, fcSize))
	model:add(nn.Dropout(0.5))
	model:add(nn.Linear(fcSize, 8))
	
	-- Add a criterion
	criterion = nn.MSECriterion() -- Mean squred error criterion

	--[[
	if false then
		print("Size of the model: " .. model:size())

		for i = 1, encoder:size() do
			local params = encoder:get(i):parameters()
			if params then
				print("Layer: " .. i)
				print("Weight size: " .. #params[1]:nElement())
				print("Bias size: " .. #params[2]:nElement())
			end
		end
	end
	--]]

	return model
end
