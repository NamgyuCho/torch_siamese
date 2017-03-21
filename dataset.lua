-- This script loads the dataset: training data and test data
-- For each set, preprocessing, such as normalization, is adopted
-- Inspired by https://github.com/alykhantejani/siamese_network/blob/master/dataset.lua

require "paths"
require "torch"

massa = {}
massa.remote_path = "https://s3-ap-southeast-1.amazonaws.com/remotesensingdata/"
massa.root_folder = "data/massa.t7"
massa.trainset_path = paths.concat(massa.root_folder, "train.t7")
massa.testset_path = paths.concat(massa.root_folder, "test.t7")

------------------------------------------------------------
-- Download the dataset
------------------------------------------------------------
function massa.download (dataset)
	if not paths.filep(massa.trainset_path) or not paths.filep(massa.testset_path) then
		local tarfile = paths.basename(massa.remote_path)
		-- download the dataset file, untar it and then remove it
		os.execute("wget " .. massa.remote_path .. "; " .. "tar xvf " .. tarfile .. "; rm " .. tarfile)
	end
end
------------------------------------------------------------


------------------------------------------------------------
-- Normalize the dataset
------------------------------------------------------------
function massa.load_normalized_dataset (filename, mean_, std_)
	local file = torch.load(filename, "ascii")

	local dataset = {}
	dataset.data = file.data:type(torch.getdefaulttensortype())
	dataset.labels = file.labels

	local std = std_ or dataset.data:std()
	local mean = mean_ or dataset.data:mean()
	dataset.data:add(-mean)
	dataset.data:mul(1.0/std)

	dataset.std = std
	dataset.mean = mean

	function dataset:size()
		return dataset.data:size(1)
	end

	local class_count = 0
	local classes = {}
	for i = 1, dataset.labels:size(1) do
		if classes[dataset.labels[i]] == nil then
			class_count = class_count + 1
			table.insert(classes, dataset.labels[i])
		end
	end

	dataset.class_count = class_count

	-- The dataset has to be indexable by the [] operator so this next bit handles that
	setmetatable(dataset, {__index = function(self, index)
		local input = self.data[index]
		local label_vector = self.labels[index]
		local example = {input, label_vector}
		return example
	end })
	
	return dataset
end
------------------------------------------------------------


------------------------------------------------------------
-- Load the dataset subset
------------------------------------------------------------
function massa.load_siamese_dataset_subset (filename, subset_size)
	local file = torch.load(filename)
	-- data structure: {
	-- diffNorm: 20
	-- imSrc: 368x3x300x300
	-- imTar: 368x3x300x300
	-- labels: 368x8
	-- }

	local all_dataSrc = file.imSrc
	local all_dataTar = file.imTar
	local all_labels = file.labels

	local sizeData = all_dataSrc:size()[1]

	if subset_size > 0 then
		sizeData = subset_size
		print("Use the subset of size: " .. sizeData)
	end

	--[[
	local dataSrc = torch.Tensor(sizeData, all_dataSrc:size()[2], all_dataSrc:size()[3], all_dataSrc:size()[4])
	local dataTar = torch.Tensor(sizeData, all_dataSrc:size()[2], all_dataSrc:size()[3], all_dataSrc:size()[4])
	local labels = torch.Tensor(sizeData, all_labels:size()[2])

	for i = 1, sizeData do
		dataSrc[i] = all_dataSrc[i]:double():mul(1.0/255.0)
		dataTar[i] = all_dataTar[i]:double():mul(1.0/255.0)
		labels[i] = all_labels[i]
	end
	--]]

	local mean1 = torch.DoubleTensor(3, 300, 300)
	local mean2 = torch.DoubleTensor(3, 300, 300)
	local std = 0

	for i = 1, sizeData do
		local src_tmp = all_dataSrc[i]:double():mul(1.0/255.0)
		local tar_tmp = all_dataTar[i]:double():mul(1.0/255.0)

		mean1 = mean1 + src_tmp + tar_tmp
		mean2 = mean2 + torch.cmul(src_tmp, src_tmp) + torch.cmul(tar_tmp, tar_tmp)
	end
	mean1:mul(1.0/(sizeData*2))
	mean2:mul(1.0/(sizeData*2))
	std = torch.sqrt(mean2 - torch.cmul(mean1, mean1))

	--[[
	print("STD")
	print(std:mean())
	print("MEAN")
	print(mean1:mean())
	print(mean1:type())
	print(mean1)
	--]]


	-- now we make the pairs (tensor of size (x, 2, 3, 300, 300) for training data)
	paired_data = torch.ByteTensor(sizeData, 2, all_dataSrc:size(2), all_dataSrc:size(3), all_dataSrc:size(4))
	paired_data_labels = torch.FloatTensor(sizeData, all_labels:size(2))
	index = 1

	for i = 1, sizeData do -- 2 for paired data
		--[[
		paired_data[i][1] = dataSrc[i]:clone()
		paired_data[i][2] = dataTar[i]:clone()
		--]]
		paired_data[i][1] = all_dataSrc[i]
		paired_data[i][2] = all_dataTar[i]
		paired_data_labels[i] = all_labels[i]
	end

	local dataset = {}
	dataset.data = paired_data
	--dataset.data:add(-mean)
	--dataset.data:mul(1.0/std)
	dataset.labels = paired_data_labels
	dataset.std = std:mean()
	dataset.mean = mean1:mean()

	--dataset.data = dataset.data:float()
	--dataset.labels = dataset.labels:float()

	function dataset:size()
		return dataset.data:size(1)
	end

	-- The dataset has to be indexable by the [] operator so this next bit handles that
	setmetatable(dataset, {__index = function (self, index)
		local input = self.data[index]
		local label = self.labels[index]
		local example = {input, label}
		return example
	end })

	return dataset
end
------------------------------------------------------------


------------------------------------------------------------
-- Load the whole dataset
------------------------------------------------------------
function massa.load_siamese_dataset (filename, subset_size)
	return massa.load_siamese_dataset_subset(filename, subset_size)
end
------------------------------------------------------------


--[[
if not paths.dirp("data") then
	os.execute("mkdir -p " .. train_dir)
	os.execute("cd " .. train_dir)

	print(sys.COLORS.red .. "<data> downloading dataset")
	--os.execute("wget " .. www .. img_file)
	--os.execute("tar -xvf " .. img_file)
else
	print(sys.COLORS.red .. "<data> using the existing dataset")
end


------------------------------------------------------------
-- load or generate new dataset:

if paths.filep("train.t7") and paths.filep("test.t7") then
	print(sys.COLORS.red .. "<data> loading previously generated dataset:")
	trainData = torch.load("train.t7")
	testData = torch.load("test.t7")

	trSize = trainData:size(1)
	teSize = testData:size(1)
else
	print(sys.COLORS.red .. "<data> creating a new dataset from files:")

	-- load files in directory
	require "paths"

	currentPath = paths.cwd() -- Current Working Directory
	local trainDir = "./imgs"

	files = {}

	for file in paths.files(trainDir) do
		if file:find("jpg" .. "$") then
			table.insert(files, paths.concat(trainDir, file))
		end
	end

	local trainImgNumber = #files

	-- sort file names
	table.sort(files, function(a, b) return a < b end)
	print("Found files:")
	print(files)

	-- load images
end
--]]

