@echo off
set MSVC_DIR=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207
set SDK_DIR=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0
set INCLUDE=%MSVC_DIR%\include;%SDK_DIR%\ucrt;%SDK_DIR%\shared;%SDK_DIR%\um;%SDK_DIR%\winrt;%SDK_DIR%\cppwinrt
set PATH=%MSVC_DIR%\bin\Hostx64\x64;%PATH%
set SDK_LIB_DIR=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0
set LIB=%MSVC_DIR%\lib\atlmfc\lib\x64;%MSVC_DIR%\lib\x64;%SDK_LIB_DIR%\ucrt\x64;%SDK_LIB_DIR%\um\x64
cl /EHsc /O2 /std:c++17 orderbook.cpp /Fe:cpp_orderbook_worker.exe