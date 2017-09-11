require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'image'

require 'nnex' -- for the custom SeparableConvolution layer

torch.setdefaulttensortype('torch.FloatTensor') -- the custom layer currently only supports single precision

cutorch.setDevice(1) -- change this if you have a multiple graphics cards and you want to utilize them

----------------------------------------------------------

local objectCommandline = torch.CmdLine()

objectCommandline:option('-model', 'lf') -- which model to use, l1 or lf, please see our paper for more details
objectCommandline:option('-first', './images/first.png') -- path to the first frame
objectCommandline:option('-second', './images/second.png') -- path to the second frame
objectCommandline:option('-out', './result.png') -- path to where the output should be stored

local objectArguments = objectCommandline:parse(arg)

----------------------------------------------------------

local moduleNetwork = torch.load('./network-' .. objectArguments['model'] .. '.t7')

----------------------------------------------------------

local tensorInputFirst = image.load(objectArguments['first'], 3, 'float'):index(1, torch.LongTensor({ 3, 2, 1 }))
local tensorInputSecond = image.load(objectArguments['second'], 3, 'float'):index(1, torch.LongTensor({ 3, 2, 1 }))
local tensorOutput = torch.FloatTensor():zero()

assert(tensorInputFirst:size(2) == tensorInputSecond:size(2))
assert(tensorInputFirst:size(3) == tensorInputSecond:size(3))

local intWidth = tensorInputFirst:size(3)
local intHeight = tensorInputFirst:size(2)

assert(intWidth <= 1280) -- while our approach works with larger images, we do not recommend it unless you are aware of the implications
assert(intHeight <= 720) -- while our approach works with larger images, we do not recommend it unless you are aware of the implications

local intPaddingLeft = math.floor(51 / 2.0)
local intPaddingTop = math.floor(51 / 2.0)
local intPaddingRight = math.floor(51 / 2.0)
local intPaddingBottom = math.floor(51 / 2.0)
local modulePaddingFirst = nn.Sequential()
local modulePaddingSecond = nn.Sequential()
local modulePaddingOutput = nn.Sequential()

if true then
	local intPaddingWidth = intPaddingLeft + intWidth + intPaddingRight
	local intPaddingHeight = intPaddingTop + intHeight + intPaddingBottom

	if intPaddingWidth ~= bit.lshift(bit.rshift(intPaddingWidth, 7), 7) then
		intPaddingWidth = bit.lshift(bit.rshift(intPaddingWidth, 7) + 1, 7) -- more than necessary
	end

	if intPaddingHeight ~= bit.lshift(bit.rshift(intPaddingHeight, 7), 7) then
		intPaddingHeight = bit.lshift(bit.rshift(intPaddingHeight, 7) + 1, 7) -- more than necessary
	end

	intPaddingWidth = intPaddingWidth - (intPaddingLeft + intWidth + intPaddingRight)
	intPaddingHeight = intPaddingHeight - (intPaddingTop + intHeight + intPaddingBottom)

	modulePaddingFirst:add(nn.SpatialReplicationPadding(intPaddingLeft, intPaddingRight + intPaddingWidth, intPaddingTop, intPaddingBottom + intPaddingHeight))
	modulePaddingSecond:add(nn.SpatialReplicationPadding(intPaddingLeft, intPaddingRight + intPaddingWidth, intPaddingTop, intPaddingBottom + intPaddingHeight))
	modulePaddingOutput:add(nn.SpatialReplicationPadding(0 - intPaddingLeft, 0 - intPaddingRight - intPaddingWidth, 0 - intPaddingTop, 0 - intPaddingBottom - intPaddingHeight))

	modulePaddingFirst = modulePaddingFirst:cuda()
	modulePaddingSecond = modulePaddingSecond:cuda()
	modulePaddingOutput = modulePaddingOutput:cuda()
end

if true then
	tensorInputFirst = tensorInputFirst:cuda()
	tensorInputSecond = tensorInputSecond:cuda()
	tensorOutput = tensorOutput:cuda()
end

if true then
	local tensorPaddingFirst = modulePaddingFirst:forward(tensorInputFirst:view(1, 3, intHeight, intWidth))
	local tensorPaddingSecond = modulePaddingSecond:forward(tensorInputSecond:view(1, 3, intHeight, intWidth))
	local tensorPaddingOutput = modulePaddingOutput:forward(moduleNetwork:forward({ tensorPaddingFirst, tensorPaddingSecond })[1])

	tensorOutput:resize(3, intHeight, intWidth):copy(tensorPaddingOutput)
end

if true then
	tensorInputFirst = tensorInputFirst:float()
	tensorInputSecond = tensorInputSecond:float()
	tensorOutput = tensorOutput:float()
end

image.save(objectArguments['out'], tensorOutput:clamp(0.0, 1.0):index(1, torch.LongTensor({ 3, 2, 1 })))