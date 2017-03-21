require "nn"
require "dataset"
require "gnuplot"
require "image"

function get_label (one_hot_vector)
	for i = 1, one_hot_vector:size()[1] do
		if one_hot_vector[i] == 1 then
			return i
		end
	end

	return 0
end

--------------------------------------------------------------
-- Parse command line options
--------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text("Argument")
cmd:argument("-test_data", "test data file (.t7)")
cmd:argument("-dataset_attributes", "dataset attributes (mean & std)")
cmd:argument("-pretrained_model", "pretrained model (.net)")
cmd:text("Options")
cmd:option("-out", "out.png", "out file (.png)")
cmd:option("-gpu", false, "use gpu")
cmd:option("-log", "", "output log file")

params = cmd:parse(arg)

if params.log ~= "" then
	cmd:log(params.log, params)
end
--------------------------------------------------------------


--------------------------------------------------------------
-- Load test data
--------------------------------------------------------------
cmd:text("")
dataset_attributes = torch.load(params.dataset_attributes)
test_data = massa.load_normalized_dataset(params.test_data, dataset_attributes.mean, dataset_attributes.std)
model = torch.load(params.pretrained_model)


--------------------------------------------------------------


--------------------------------------------------------------
-- Parse command line options
--------------------------------------------------------------

--------------------------------------------------------------


--------------------------------------------------------------
-- Parse command line options
--------------------------------------------------------------

--------------------------------------------------------------





--------------------------------------------------------------
--------------------------------------------------------------
--------------------------------------------------------------
--------------------------------------------------------------
--------------------------------------------------------------
--------------------------------------------------------------
--------------------------------------------------------------
--------------------------------------------------------------
--------------------------------------------------------------
--------------------------------------------------------------
--------------------------------------------------------------
