local SeparableConvolution, parent = torch.class('nn.SeparableConvolution', 'nn.Module')

function SeparableConvolution:__init()
	parent.__init(self)

	if true then
		self.output = torch.Tensor()
		self.gradInput = {}
		self.gradInput[1] = torch.Tensor()
		self.gradInput[2] = torch.Tensor()
		self.gradInput[3] = torch.Tensor()
	end
end

function SeparableConvolution:updateOutput(input)
	local intBatches = input[1]:size(1)
	local intInputDepth = input[1]:size(2)
	local intInputHeight = input[1]:size(3)
	local intInputWidth = input[1]:size(4)
	local intFilterSize = math.min(input[2]:size(2), input[3]:size(2))
	local intOutputHeight = math.min(input[2]:size(3), input[3]:size(3))
	local intOutputWidth = math.min(input[2]:size(4), input[3]:size(4))

	assert(intInputHeight - 51 == intOutputHeight - 1)
	assert(intInputWidth - 51 == intOutputWidth - 1)
	assert(intFilterSize == 51)

	assert(input[1]:isContiguous() == true)
	assert(input[2]:isContiguous() == true)
	assert(input[3]:isContiguous() == true)

	self.output:resize(intBatches, intInputDepth, intOutputHeight, intOutputWidth):zero()

	if torch.typename(input[1]):find('torch.Cuda') ~= nil then
		input[1].nn.SeparableConvolution_cuda_forward(self, input[1], input[2], input[3], self.output)

	elseif torch.typename(input[1]):find('torch.Cuda') == nil then
		assert(false) -- NOT IMPLEMENTED

	end

	return self.output
end

function SeparableConvolution:updateGradInput(input, gradOutput)
	assert(false) -- NOT IMPLEMENTED
end

function SeparableConvolution:clearState()
	self.output:set()
	self.gradInput[1]:set()
	self.gradInput[2]:set()
	self.gradInput[3]:set()

	return self
end