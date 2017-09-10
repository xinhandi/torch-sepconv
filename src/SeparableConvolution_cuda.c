#include <THC.h>
#include <THCGeneral.h>

#include "SeparableConvolution_kernel.h"

int SeparableConvolution_cuda_forward(lua_State* state) {
	SeparableConvolution_kernel_forward(
		getTorchState(state),
		(THCudaTensor*) luaT_checkudata(state, 2, "torch.CudaTensor"),
		(THCudaTensor*) luaT_checkudata(state, 3, "torch.CudaTensor"),
		(THCudaTensor*) luaT_checkudata(state, 4, "torch.CudaTensor"),
		(THCudaTensor*) luaT_checkudata(state, 5, "torch.CudaTensor")
	);

	return 1;
}

const struct luaL_Reg SeparableConvolution_cuda_register[] = {
	{ "SeparableConvolution_cuda_forward", SeparableConvolution_cuda_forward },
	{ NULL, NULL }
};

void SeparableConvolution_cuda_init(lua_State* state) {
	luaT_pushmetatable(state, "torch.CudaTensor");
	luaT_registeratname(state, SeparableConvolution_cuda_register, "nn");
	lua_pop(state, 1);
}