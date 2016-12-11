require 'torch'
N = 5
local A =torch.rand(N)
do
   local A = torch.rand(N, N)
   print(A)
end
A = torch.rand(N, N)
b = torch.rand(N)

function J(x)
return 0.5*x:dot(A*x)-b:dot(x)
end

function dJ(x)
 return A*x-b
end

do
   local neval = 0
   function JdJ(x)
      local Jx = J(x)
      neval = neval+1
      print(string.format('after %d evaluations J(x) = %f', neval, Jx))
      return Jx, dJ(x)
   end
end

require 'optim'
state = {
   verbose = true,
   maxIter = 100
}
X = torch.rand(N)
optim.cg(JdJ, X, state)

evaluations = {}
time = {}
timer = torch.Timer()
neval = 0
function JdJ(x)
   local Jx = J(x)
   neval = neval +1
   print(string.format('after %d evaluations, J(x) = %f', neval, Jx))
   table.insert(evaluations, Jx)
   table.insert(time, timer:time().real)
   return Jx, dJ(x)
end

x0 = torch.rand(N)
cgx = x0:clone()
timer:reset()
optim.cg(JdJ, cgx, state)
cgtime = torch.Tensor(time)
cgevaluations = torch.Tensor(evaluations)

evaluations = {}
time = {}

state = {
   lr = 0.1
}
x = x0:clone()
timer:reset()
for i=1,1000 do
   optim.sgd(JdJ, x, state)
   table.insert(evaluations, Jx)
end
sgdtime = torch.Tensor(time)
sgdevaluations = torch.Tensor(evaluations)

require 'gnuplot'

gnuplot.figure(1)
gnuplot.title('CG loss minimisation over time')
gnuplot.plot(cgtime, cgevaluations)

gnuplot.figure(2)
gnuplot.title('SGD minimisation over time')
gnuplot.plot(sgdtime, sgdevaluations)

gnuplot.pngfigure('plot.png')
gnuplot.plot(
   {'CG',  cgtime,  cgevaluations,  '-'},
   {'SGD', sgdtime, sgdevaluations, '-'})
gnuplot.xlabel('time (s)')
gnuplot.ylabel('J(x)')
gnuplot.plotflush()
