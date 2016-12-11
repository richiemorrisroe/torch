
require 'rnn'

r = nn.Recurrent(7,
                  nn.LookupTable(10, 7),
                  nn.Linear(7, 7),
                  nn.Sigmoid(),
                  5
)

rm = nn.Sequential():add(nn.ParallelTable()
           :add(nn.LookupTable(10, 7))
           :add(nn.Linear(7, 7))):add(nn.CAddTable()):add(nn.Sigmoid())

r = nn.Recurrence(rm, 7, 1)

rr = nn.Sequential():add(r):add(nn.Linear(7, 10)):add(nn.LogSoftMax())

rnn == nn.Recursor(rr, 5)

outputs, err = {}, 0
criterion = nn.ClassNLLCriterion()
for step=1,5  do
   outputs[step] = rnn:forward(inputs[step])
   err = err + criterion:forward(outputs[step], targets[step])
end

gradOutputs, gradInputs = {}, {}
for step 5, 1,-1 do
   gradOutputs[step] = criterion:backward(outputs[step], targets[step])
   gradInputs[step] = rnn:backward(inputs[step], gradOutputs[step])
end

rnn:updateParameters(0.1)
rnn:forget()

rnn:zeroGradParameters()

rnn = nn.Sequencer(rr)
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

outputs = rnn:forward(inputs)
err = criterion:forward(outputs, targets)
gradOutputs = criterion:backward(outputs, targets)
gradInputs = rnn:backward(inputs, gradOutputs)
rnn:updateParameters(0.1)
rnn:zeroGradParameters()
rr:add(nn.NormStablizer())
