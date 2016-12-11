dataset = {}
function dataset:size(n) return n end
for i=1,dataset:size() do
   local input = torch.randn(2)
   local output = torch.Tensor(1)
   if input[1]*input[2]>0 then
      output[1] = -1
   else
      output[1] = 1
   end
   dataset[i] = {input, output}
end

require 'nn'

mlp = nn.Sequential()
inputs = 2; outputs =1; HUs=20
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, outputs))

criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate= 0.01
trainer:train(dataset)
criterion = nn.MSECriterion()


for i = 1,2500 do
   local input torch.randn(2)
   local output torch.Tensor(1)
   if input[1]*input[2]>0 then
      output[1] = -1
   else
      output[1] = 1
   end
   criterion:forward(mlp:forward(input), output)
   mlp:zeroGradParameters()
   mlp:backward(input, criterion:backward(mlp.output, output))
   mlp:updateParameters(0.01)
end
