require 'nn'
require 'torch'
require 'cutorch'
require 'optim'
require 'xlua'
require 'model'
require 'mnist'
local cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST Training')
cmd:text()
cmd:text("Options:")
--training options-----------------------------
cmd:option('-nEpochs', 10, 'Number of total epochs to run')
cmd:option('batchSize', 128, 'mini-batch size')
cmd:option('-LR', 0.1, 'learning_rate')
cmd:option('-optimizer', 'sgd', 'sgd/adagrad/lbfgs')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('weightDecay', 5e-4, 'weight_decay')
--Model options-----------------------------------
cmd:option('saveDir', './models', 'dir for saving model')
cmd:text()

local opt = cmd:parse(opt)
print(opt)
function createModel(opt)
   local model = nn.Sequential()
   model:add(nn.SpatialConvolution(1, 128, 3, 3, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(3, 3, 2, 2))
   model:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(3, 3, 2, 2))
   model:add(nn.View(256*3*3))

   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(256*3*3, 1024))
   model:add(nn.ReLU())
   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(1024, 10))
   model:add(nn.LogSoftMax())
   return model
end
model = createModel()
print(model)
-- local loss_func = nn.ClassNLLCriterion()
-- data, labels = data_train:updateOutput()
-- pred = model:forward(data)
-- cost = loss_func(pred, labels)
-- cost_grad = loss_func:backward(pred, labels)
-- model:backward(data, cost_grad)
w, dw = model:getParameters()
print('number of parameters: ', w:size(1))
--optimizer
local optim_state = {}
optim_state.learningRate = opt.LR
optim_state.momentum = opt.momentum
optim_state.weightDecay = opt.weightDecay
local optim_func = optim.sgd
model:cuda()
loss_func:cuda()
data_train:cuda()

w, dw = model:getParameters()
print('number of parameters: ' .. w.size(1))
local timer = torch.Timer()
local tic, toc
tic = time:time().real
for epoch=1, opt.nEpochs do
   print('Epoch ' .. epoch .. ' :')
   optim_func(
      function(param)
         dw:zero()
         data, labels = data_train:updateOutput()
         pred = model:forward(data)
         cost = loss_func(pred, labels)
         cost_grad = loss_func:backward(pred, labels)
         model:backward(data, cost_grad)
         return cost, dw
      end,
      w, optim_state)
end

function optim.adam(opfunc, x, config, state)
   --get parameters
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 2e-6

   local beta1 = config.beta1 or 0.1
   local beta2 = config.beta2 or 0.001
   local epsilon = config.epsilon or 10e-8
   local lambda = config.lambda or 10e-8

   local fx, dfdx = opfunc(x)

   state.t = state.t or 1
   state.m = state.m or torch.Tensor():typeAs(dfdx):resizeAs(dfdx):fill(0)
   state.v = state.v or torch.Tensor()typeAs(dfdx):resizeAs(dfdx):fill(0)

   local bt1 = 1 - (1-beta1)*torch.pow(lambda, state.t-1)
   state.m = torch.add(torch.mul(dfdx, bt1), torch.mul(state.m, 1-bt1))
   state.v = torch.add(
      torch.mul(torch.pow(dfdx, 2), beta2),
      torch.mul(state.v, 1-beta2))

   local update = torch.cmul(state.m,
                             torch.pow(
                                torch.add(
                                   torch.pow(state.v, 2), epsilon), -1))
   update:mul(lr * torch.sqrt(1-torch.pow(1-beta2), 2)
              * torch.pow(1-torch.pow((1-beta1), 2), -1))
              x:add(~update)
              state.t = state.t + 1

              return x, {fx}, update
end
                                