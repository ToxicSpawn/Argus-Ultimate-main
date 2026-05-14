# Find all language runtime paths on this Windows machine
$checks = @(
    @{name="ruby";    paths=@("C:\Ruby34-x64\bin\ruby.exe","C:\Ruby33-x64\bin\ruby.exe","C:\tools\ruby34\bin\ruby.exe","C:\ProgramData\chocolatey\bin\ruby.exe")},
    @{name="node";    paths=@("C:\Program Files\nodejs\node.exe","C:\nvm\versions\node\v22.14.0\node.exe","C:\ProgramData\nvm\v22.14.0\node.exe","C:\ProgramData\chocolatey\bin\node.exe")},
    @{name="kotlinc"; paths=@("C:\ProgramData\chocolatey\lib\kotlin\tools\bin\kotlinc.bat","C:\ProgramData\chocolatey\bin\kotlinc.bat","C:\kotlin\bin\kotlinc.bat")},
    @{name="crystal"; paths=@("C:\ProgramData\chocolatey\bin\crystal.exe","C:\crystal\crystal.exe","C:\tools\crystal\crystal.exe","C:\ProgramData\chocolatey\lib\crystal\tools\crystal.exe")},
    @{name="Rscript"; paths=@("C:\Program Files\R\R-4.5.2\bin\Rscript.exe","C:\Program Files\R\R-4.5.1\bin\Rscript.exe")},
    @{name="elixir";  paths=@("C:\ProgramData\chocolatey\lib\elixir\tools\bin\elixir.bat","C:\Program Files\Elixir\bin\elixir.bat")},
    @{name="julia";   paths=@("C:\Users\hinge\AppData\Local\Programs\julia-1.12.0\bin\julia.exe","C:\julia\bin\julia.exe","C:\ProgramData\chocolatey\lib\julia\tools\bin\julia.exe")},
    @{name="ghc";     paths=@("C:\tools\ghc-9.8.2\bin\ghc-9.8.2.exe","C:\tools\ghc-9.8.2\bin\ghc.exe")}
)
foreach ($check in $checks) {
    $found = $false
    foreach ($p in $check.paths) {
        if (Test-Path $p) {
            Write-Host "FOUND $($check.name): $p"
            $found = $true
            break
        }
    }
    if (-not $found) { Write-Host "MISSING $($check.name)" }
}
