# Find actual executable paths for languages not in PATH
$targets = @{
    "clojure" = @("C:\ProgramData\chocolatey\lib\clojure\tools\clojure\bin\clojure.ps1",
                  "C:\tools\Clojure\clojure.ps1",
                  "C:\ProgramData\chocolatey\bin\clojure.ps1")
    "node"    = @("C:\Program Files\nodejs\node.exe",
                  "C:\nvm\v22.14.0\node.exe",
                  "C:\nvm\v20.18.0\node.exe",
                  "C:\ProgramData\nvm\v22.14.0\node.exe")
    "ruby"    = @("C:\Ruby34-x64\bin\ruby.exe","C:\Ruby33-x64\bin\ruby.exe",
                  "C:\RailsInstaller\Ruby2.3.3\bin\ruby.exe")
    "Rscript" = @("C:\Program Files\R\R-4.5.2\bin\Rscript.exe",
                  "C:\Program Files\R\R-4.5.1\bin\Rscript.exe")
    "julia"   = @("C:\julia-1.11.3\bin\julia.exe",
                  "C:\julia-1.10.0\bin\julia.exe",
                  "C:\Users\hinge\AppData\Local\Programs\julia-1.11.0\bin\julia.exe",
                  "C:\Users\hinge\AppData\Local\Programs\julia-1.10.0\bin\julia.exe",
                  "C:\julia\bin\julia.exe")
    "elixir"  = @("C:\ProgramData\chocolatey\lib\elixir\tools\bin\elixir.bat",
                  "C:\Program Files\Elixir\bin\elixir.bat")
    "crystal" = @("C:\crystal\crystal.exe",
                  "C:\Users\hinge\Downloads\Argus-Ultimate-main\Argus-Ultimate-main\crystal\crystal.exe",
                  "C:\tools\crystal\crystal.exe")
    "kotlinc" = @("C:\kotlin\bin\kotlinc.bat",
                  "C:\kotlinc\bin\kotlinc.bat",
                  "C:\ProgramData\chocolatey\lib\kotlin\tools\bin\kotlinc.bat")
    "erl"     = @("C:\Program Files\Erlang OTP\bin\erl.exe")
    "escript" = @("C:\Program Files\Erlang OTP\bin\escript.exe",
                  "C:\ProgramData\chocolatey\bin\escript.exe")
}
foreach ($name in $targets.Keys | Sort-Object) {
    $found = $false
    foreach ($p in $targets[$name]) {
        if (Test-Path $p) {
            Write-Host "FOUND $($name.PadRight(10)) $p"
            $found = $true
            break
        }
    }
    if (-not $found) { Write-Host "MISS  $($name.PadRight(10))" }
}
# Also search for julia specifically
Write-Host "`nSearching for julia.exe..."
Get-ChildItem "C:\Users\hinge\AppData\Local\Programs" -Recurse -Filter "julia.exe" -ErrorAction SilentlyContinue | Select-Object FullName
Get-ChildItem "C:\" -Recurse -Depth 4 -Filter "julia.exe" -ErrorAction SilentlyContinue | Select-Object FullName
