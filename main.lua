require "torch"
require "nn"
require "optim"
require "image"
require "dataset"
require "model"
require "paths"
require "math"

-------------------------------------------------------
-- parse command line options
-------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text("Arguments")
cmd:argument("-training_data", "training data (.t7) file")
cmd:argument("-max_epochs", "maximum epochs")
cmd:text("Options")
cmd:option("-batch_size", 50, "batch size")
cmd:option("-learning_rate", 0.001, "learning rate")
cmd:option("-momentum", 0.9, "momentum")
cmd:option("-snapshot_dir", "./snapshot", "snapshot directory")
cmd:option("-snapshot_epoch", 0, "snapshot after how many iteratinos?")
cmd:option("-gpu", false, "use gpu")
cmd:option("-weights", "", "pretrained model to begin training from")
cmd:option("-criterion", "", "criterion of the pretrained to be loaded as 'weights'")
cmd:option("-log", "output log file")
cmd:option("-dataset_size", 0, "the size of a target dataset to be used for training (0 = full size)")
cmd:option("-threads", 1, "number of threads")
cmd:option("-seed", 1, "fixed input seed for repeatable experiments")

params = cmd:parse(arg)
-------------------------------------------------------


-------------------------------------------------------
-- Initialize variable
-------------------------------------------------------
if params.log ~= "" then
	cmd:log(params.log, params)
	cmd:addTime("torch_benchmarks", "%F %T")
	print("setting log file as " .. params.log)
end

libs = {}
run_on_gpu = false
if params.gpu then
	print("using cudnn")
	require "cudnn"
	require "cutorch"
	require "cunn"

	libs["SpatialConvolution"] = cudnn.SpatialConvolution
	libs["SpatialMaxPooling"] = cudnn.SpatialMaxPooling
	libs["ReLU"] = cudnn.ReLU
	run_on_gpu = true
else
	libs["SpatialConvolution"] = nn.SpatialConvolution
	libs["SpatialMaxPooling"] = nn.SpatialMaxPooling
	libs["ReLU"] = nn.ReLU
end
torch.setnumthreads(params.threads)
torch.manualSeed(params.seed)

epoch = 0
batch_size = params.batch_size
train_error = 0

-- Load model and criterion
if params.weights ~= "" then
	print(sys.COLORS.red .. "loading model from pretrained weights in file " .. params.weights)
	model = torch.load(params.weights)
	print(sys.COLORS.red .. "loading model criterion in file " .. params.criterion)
	criterion = torch.load(params.criterion)
	--criterion = nn.MSECriterion()
else
	print(sys.COLORS.red .. "Load a new model")
	model = build_model(libs)
end

if run_on_gpu then
	model = model:cuda()
	criterion = criterion:cuda()
end

print(model)
print(criterion)

-- retrieve a view (same memory) of the parameters and gradients of these (wrt loss) of the model (Global)
model_params, grad_params = model:getParameters()
-------------------------------------------------------


-------------------------------------------------------
-- Traning function
-------------------------------------------------------
function train (data)
	local saved_criterion = false

	for i = 1, params.max_epochs do
		train_one_epoch(data)

		if params.snapshot_epoch > 0 and (epoch % params.snapshot_epoch) == 0 then -- epoch is global
			local filename = paths.concat(params.snapshot_dir, "snapshot_epoch_" .. epoch .. ".net")
			os.execute("mkdir -p " .. sys.dirname(filename))
			torch.save(filename, model)
			-- must save std, mean and criteron?

			if not saved_criterion then
				local criterion_filename = paths.concat(params.snapshot_dir, "_criterion.net")
				torch.save(criterion_filename, criterion)

				local dataset_attributes_filename = paths.concat(params.snapshot_dir, "_dataset.params")
				dataset_attributes = {}
				dataset_attributes.mean = data.mean
				dataset_attributes.std = data.std
				torch.save(dataset_attributes_filename, dataset_attributes)
			end
		end
	end
end

function train_one_epoch (dataset)
	local time = sys.clock()

	-- add random shuffling here
	shuffle = torch.randperm(dataset:size())

	for mini_batch_start = 1, dataset:size(), batch_size do -- for each mini-batch
		local inputs = {}
		local labels = {}

		-- create a mini_batch
		for i = mini_batch_start, math.min(mini_batch_start + batch_size - 1, dataset:size()) do
			local input = dataset[shuffle[i]][1]:clone() -- the tensor containing two images
			local label = dataset[shuffle[i]][2]:clone()

			input = input:float():mul(1.0/255.0)
			input[1]:add(-dataset.mean)
			input[1]:mul(1.0/dataset.std)
			input[2]:add(-dataset.mean)
			input[2]:mul(1.0/dataset.std)

			if run_on_gpu then
				input = input:cuda()
				label = label:cuda()
			end

			table.insert(inputs, input)
			table.insert(labels, label)
		end
		
		-- create a closure to evaluate df/dX where x are the model model_params at a given point
		-- and df/dX is the gradient of the loss wrt to these model_params
		local func_eval =
		function (x)
			-- update the model model_params (copy x in to model_params)
			if x ~= model_params then
				model_params:copy(x)
			end

			grad_params:zero() -- reset gradients

			local avg_error = 0 -- the average error of all criterion outs

			--print("constructing a mini batch")

			-- evaluate for complete mini_batch
			for i = 1, #inputs do
				local output = model:forward(inputs[i])
				local err = criterion:forward(output, labels[i])

				avg_error = avg_error + err

				-- estimate dLoss/dW
				local dloss_dout = criterion:backward(output, labels[i])
				model:backward(inputs[i], dloss_dout)
			end
			
			grad_params:div(#inputs)
			avg_error = avg_error / #inputs

			if train_error == 0 then
				train_error = avg_error
			else
				train_error = (train_error + avg_error) / 2
			end

			return avg_error, grad_params
		end

		config = {
			learningRate = params.learning_rate, 
			momentum = params.momentum}

		-- This function updates the global model_params variable (which is a view on the models model_params)
		if run_on_gpu then
			--model_params = model_params:cuda()
		end

		optim.sgd(func_eval, model_params, config)

		xlua.progress(mini_batch_start, dataset:size()) -- display progres
	end

	-- time taken
	time = sys.clock() - time
	print("Time taken for 1 epoch = " .. (time * 1000 * 1000 / 60) .. "mins, time taken to learn 1 sample = " .. ((time/dataset:size()) * 1000) .. "ms")
	epoch = epoch + 1

	print(sys.COLORS.red .. string.format("<Train error> avgerage error at epoch %d = %f", epoch, train_error))
end
-------------------------------------------------------

currentPath = paths.cwd() -- Current Working Directory
fileName = paths.concat(currentPath, params.training_data)

print(sys.COLORS.red .. "\n<Loading dataset...>")
massa_dataset = massa.load_siamese_dataset(params.training_data, params.dataset_size)
print(sys.COLORS.red .. "<Dataset loaded>")
print(massa_dataset)

print(sys.COLORS.red .. "\n<Start training>")
train(massa_dataset)

