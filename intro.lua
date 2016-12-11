require 'torch'
require 'nn'
a = torch.Tensor(5, 3)
a = torch.rand(5, 3)
b = torch.rand(3, 4)
--matrix mult
c =a*b
torch.mm(a, b)
c= torch.Tensor(5, 4)
c:mm(a, b)
require 'cutorch'
a = a:cuda()
b = b:cuda()
c = c:cuda()
--done on GPU
c:mm(a, b)

function addTensors(a, b)
   return a+b
end

a = torch.ones(5, 2)
b = torch.Tensor(2,5):fill(4)
print(addTensors(a, b))

require 'nn'

net = nn.Sequential()
-- 1 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.SpatialConvolution(1, 6, 5, 5))
-- a max pooling operation that looks at 2x2 windows
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net:add(nn.SpatialConvolution(6, 16, 5, 5))
--reshapes from a 3d tensor of 16*5*5 to a 1d tensor of 16*5*5
net:add(nn.View(16*5*5))
-- fully connected layer - mat mul between input and weights
net:add(nn.Linear(16*5*5, 120))
net:add(nn.Linear(120, 84))
--10 is the number of outputs of the network
net:add(nn.Linear(84, 10))
net:add(nn.LogSoftMax())

print( 'Lenet5\n' .. net:__tostring())

input = torch.rand(1, 32, 32)
output = net:forward(input)
--zero the internal gradient buffer, to be continued
net:zeroGradParameters()
gradInput = net:backward(input, torch.rand(10))
-- a negative log likelihood criterion for multi-class classification
criterion = nn.ClassNLLCriterion()
criterion:forward(output, 3)
gradients = criterion:backward(output, 3)
m = nn.SpatialConvolution(1, 3, 2, 2)
print(m.weight)
print(m.bias)
os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
os.execute('unzip cifar10torchsmall.zip')
trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

setmetatable(trainset,
             {__index = function(t, i)
                 return {t.data[i], t.label[i]}
             end})
trainset.data = trainset.data:double()
function trainset:size()
   return self.data:size(1)
end

redChannel = trainset.data[{{}, {1}, {}, {}}]
print(#redChannel)
mean = {}
stdv = {}
for i=1,3 do
   mean[i] = trainset.data[{{}, {i}, {}, {}}]:mean()
   -- print('Channel', .. i .. ', Mean: ' .. mean[i])
   trainset.data[{{}, {i}, {}, {}}]:add(-mean[i])
   stdv[i] = trainset.data[{{}, {i},{},{}}]:std()
   -- print('Channel', .. i .. ', Stand Dev: ' .. stdv[i])
   trainset.data[{ {}, {i}, {}, {}}]:div(stdv[i])
end

net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5))
net:add(nn.SpatialMaxPooling(2, 2, 2,2))
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net:add(nn.View(16*5*5))
net:add(nn.Linear(16*5*5, 120))
net:add(nn.Linear(120, 84))
net:add(nn.Linear(84, 10))
net:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5
trainer:train(trainset)

print(classes[testset.label[100]])
testset.data = testset.data:double()
for i=1,3 do
   testset.data[{{}, {i}, {}, {}}]:add(-mean[i])
   testset.data[{{}, {i}, {}, {}}]:div(stdv[i])
end
horse = testset.data[100]
print(horse:mean(), horse:std())
print(classes[testset.label[100]])
predicted = net:forward(testset.data[100])
--network outputs log probabilities
print(predicted:exp())
for i=1,predicted:size(1) do
   print(classes[i], predicted[i])
end

correct = 0
for i=1,10000 do
   local groundtruth = testset.label[i]
   local prediction = net:forward(testset.data[i])
   --sort in descending order
   local confidences, indices = torch.sort(prediction, true)
   if groundtruth == indices[1] then
      correct = correct +1
   end
end

print(correct, 100*correct/10000)
class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1,10000 do
   local groundtruth = testset.label[i]
   local prediction = net:forward(testset.data[i])
   local confidences, indices = torch.sort(prediction, true)
   if groundtruth == indices[1] then
      class_performance[groundtruth] = class_performance[groundtruth]+1
   end
end

for i=1,#classes do
   print(classes[i], 100*class_performance[i]/1000)
end
--now with added GPU!
require 'cunn'
net = net:cuda()
criterion = criterion:cuda()
trainset.data = trainset.data:cuda()
testset.data = testset.data:cuda()
-- trainset = trainset:cuda()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5
trainer:train(trainset)
trainset.data=trainset.data:double()
