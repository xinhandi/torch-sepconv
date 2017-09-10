#!/bin/bash

luarocks make rocks/nnex-scm-1.rockspec

wget --timestamping http://content.sniklaus.com/SepConv/network-l1.t7
wget --timestamping http://content.sniklaus.com/SepConv/network-lf.t7