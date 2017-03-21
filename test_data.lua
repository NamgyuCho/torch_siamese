require "torch"
require "nn"
require "optim"
require "image"
require "dataset"
require "paths"

print("Loading dataset...")
currentPath = paths.cwd() -- Current Working Directory
datasetName = "./data/train.t7"
fileName = paths.concat(currentPath, datasetName)
print(fileName)
massa_dataset = massa.load_siamese_dataset(fileName)
print("Dataset loaded")
print(massa_dataset)
