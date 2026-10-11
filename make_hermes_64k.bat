@echo off
rem Makes a copy of an Ollama model that is served with a 64K context window, which Hermes Agent requires.
rem It only adds a tiny new entry that reuses the model's existing files. Remove it any time with:  ollama rm <name>-64k
rem Usage:   make_hermes_64k.bat qwen3.5:4b        ->  creates  qwen3.5-4b-64k
if "%~1"=="" (
  echo Usage: %~nx0 ^<ollama-model^>      e.g.  %~nx0 qwen3.5:4b
  exit /b 1
)
set "SRC=%~1"
set "NAME=%SRC::=-%"
set "NAME=%NAME:/=-%-64k"
set "MF=%TEMP%\Modelfile.%NAME%"
> "%MF%" echo FROM %SRC%
>> "%MF%" echo PARAMETER num_ctx 65536
echo Creating %NAME% from %SRC% with a 65536-token context...
ollama create "%NAME%" -f "%MF%"
if errorlevel 1 (
  echo FAILED. Is Ollama running and is "%SRC%" installed?
  exit /b 1
)
echo.
echo Done. In the voice app: Settings -^> Agent model -^> %NAME%
echo Memory note: 64K of context needs more VRAM; with Flash Attention + q8 KV cache a 4B model fits in about 4 GB.
