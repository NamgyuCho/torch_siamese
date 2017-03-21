-- This script loads the dataset: training data and test data
-- For each set, preprocessing, such as normalization, is adopted
--
-- Inspired by https://github.com/torch/demos/blob/master/person-detector/data.lua

------------------------------------------------------------
-- load or generate new dataset:
------------------------------------------------------------
print(sys.COLORS.red .. "<data> creating a new dataset from files:")

-- load files in directory
require "paths"

currentPath = paths.cwd() -- Current Working Directory
--print("Current path: " .. currentPath)
-- Note that lua does not work well with relative path

local trainImgDir = "./data/train/image/"
local trainLabelDir = "./data/train/label/"

print("Start finding files in " .. trainLabelDir)

-- load label file names
fileLabel = {}

for file in paths.files(trainLabelDir) do
	if file:find("txt" .. "$") then
		local filename = trainLabelDir .. file
		--print(filename)
		table.insert(fileLabel, filename)
	end
end

local sizeTrain = #fileLabel

-- sort file names
table.sort(fileLabel, function(a, b) return a < b end)
--print("Found files:")
--print(fileLabel)
print("Found " .. sizeTrain .. " files.")
------------------------------------------------------------


-----------------------------------------------------------------
-- load label files
-----------------------------------------------------------------
Label = {}
cntErr = 0
cntOk = 0
for i = 1, sizeTrain do
	local f = io.open(fileLabel[i], "r")
	local labelTmp = {}

	local cntElm = 1
	if f then
		while true do
			local f_val = f:read("*n")
			if not f_val then 
				break 
			end

			if cntElm >= 9 and cntElm <= 16 then
				labelTmp[#labelTmp + 1] = f_val
			end
			cntElm = cntElm + 1
		end

		io.close(f)
		cntOk = cntOk + 1
	else
		cntErr = cntErr + 1
	end
	Label[#Label + 1] = labelTmp
end
print(cntOk .. " files are valid.")
print(cntErr .. " files are invalid.")
--torch.save("./data/train_label.t7", Label)
--print(Label[10])
-----------------------------------------------------------------


-----------------------------------------------------------------
-- construct a table containing file names
-----------------------------------------------------------------
fileImage = {}

for i = 1, sizeTrain do
	table.insert(fileImage, {src = paths.concat(trainImgDir, fileLabel[i]:sub(-13, -5) .. "_src.jpg"),
							 tar = paths.concat(trainImgDir, fileLabel[i]:sub(-13, -5) .. "_tar.jpg")})
end
--print("Found images:")
--print(fileImage)
-----------------------------------------------------------------

-- It seems that the size of the dataset is not affordable at the current OS system.
-- So, for a temporal solution, I decided to save only filenames so that they are loaded during constructing mini batches.

--print(sys.COLORS.red .. "Save the image list and label list")
--torch.save("./data/img_list.t7", fileImage)
--torch.save("./data/label_list.t7", fileLabel)


-----------------------------------------------------------------
-- Construct the training dataset structure
-----------------------------------------------------------------
sizeImWidth = 300
sizeImHeight = 300
sizeChannel = 3 -- since it is a siamese network
sizeLabel = 2 * 4 -- difference between two corner points
numIm = 2

trainData = {
	imSrc = torch.ByteTensor(sizeTrain, sizeChannel, sizeImWidth, sizeImHeight),
	imTar = torch.ByteTensor(sizeTrain, sizeChannel, sizeImWidth, sizeImHeight),
	labels = torch.Tensor(sizeTrain, sizeLabel),
	diffNorm = 20,
	size = function() return sizeTrain end
}

-- shuffle dataset: get shuffled indices in this variable:
--local idxTrainShuffle = torch.randperm(sizeTrain)

-- load train image data
require "image"

for i = 1, sizeTrain do
	local img_tmp = image.load(fileImage[i].src)
	img_tmp:mul(255):byte()
	trainData.imSrc[i] = img_tmp:clone()

	local img_tmp = image.load(fileImage[i].tar)
	img_tmp:mul(255):byte()
	trainData.imTar[i] = img_tmp:clone()

	trainData.labels[i] = torch.Tensor(Label[i])

	if i % 1000 == 0 then
		print(i .. " files processed...")
	end
end

-- save created dataset
torch.save('./data/train.t7', trainData)

-- Displaying the dataset architecture
print(sys.COLORS.red .. "Training data: ")
print(trainData)
print()

-- preprocessing
trainData.size = function() return sizeTrain end
------------------------------------------------------------
