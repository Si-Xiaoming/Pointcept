#!/bin/bash

# 设置HTTP和HTTPS代理
export http_proxy=http://127.0.0.1:1087
export https_proxy=http://127.0.0.1:1087

# 可选：设置无代理访问的地址
# export no_proxy="localhost,127.0.0.1,.example.com"

# 脚本的其余部分...
echo "代理已设置，继续执行后续操作..."