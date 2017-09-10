#include <luaT.h>
#include <lua.h>

#include <THC.h>
#include <THCGeneral.h>

THCState* getTorchState(lua_State* state) {
	THCState* torch = NULL;

	lua_getglobal(state, "cutorch");
	lua_getfield(state, -1, "getState");
	lua_call(state, 0, 1);
	torch = (THCState*) lua_touserdata(state, -1);
	lua_pop(state, 2);

	return torch;
}

#include "src/SeparableConvolution_cuda.c"

LUA_EXTERNC DLL_EXPORT int luaopen_libnnex(lua_State* state);

int luaopen_libnnex(lua_State* state) {
	lua_newtable(state);

	SeparableConvolution_cuda_init(state);

	return 1;
}