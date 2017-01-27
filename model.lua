require 'nn'

function build_model(libs)
    local SpatialConvolution = libs['SpatialConvolution']
    local SpatialMaxPooling = libs['SpatialMaxPooling']
    local ReLU = libs['ReLU']

    --Encoder / Embedding
    --Input dims are 28x28
    encoder = nn.Sequential()
    encoder:add(nn.SpatialConvolution(1, 20, 5, 5))
    encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    encoder:add(nn.SpatialConvolution(20, 50, 5, 5))
    encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    encoder:add(nn.View(50*5*5))
    encoder:add(nn.Linear(50*5*5, 500))
    encoder:add(nn.ReLU())
    encoder:add(nn.Linear(500, 10))

    -- The siamese model
    siamese_encoder = nn.ParallelTable()
    siamese_encoder:add(encoder)
    siamese_encoder:add(encoder:clone('weight', 'bias', 'gradWeight', 'gradBias')) --clone the encoder and share the weight, bias. Must also share the gradWeight and gradBias

    -- The siamese model (inputs will be Tensors of shape (2, channel, height, width))
    model = nn.Sequential()
    model:add(nn.SplitTable(1))
    model:add(siamese_encoder)
    model:add(nn.PairwiseDistance(2))

    margin = 1
    criterion = nn.HingeEmbeddingCriterion(margin)

    return model
end