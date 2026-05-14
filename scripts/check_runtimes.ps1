# Check all language runtimes available for Argus native workers
$checks = @(
    @{name="go";       cmd="go version"},
    @{name="cargo";    cmd="cargo --version"},
    @{name="java";     cmd="java --version"},
    @{name="dotnet";   cmd="dotnet --version"},
    @{name="scala";    cmd="scala --version"},
    @{name="erl";      cmd="erl -version"},
    @{name="ghc";      cmd="ghc --version"},
    @{name="cabal";    cmd="cabal --version"},
    @{name="clojure";  cmd="clojure --version"},
    @{name="node";     cmd="node --version"},
    @{name="ruby";     cmd="ruby --version"},
    @{name="Rscript";  cmd="Rscript --version"},
    @{name="julia";    cmd="julia --version"},
    @{name="elixir";   cmd="elixir --version"},
    @{name="crystal";  cmd="crystal --version"},
    @{name="kotlinc";  cmd="kotlinc -version"}
)
foreach ($c in $checks) {
    try {
        $out = (cmd /c "$($c.cmd) 2>&1") -join " "
        $short = ($out -replace '\r?\n',' ').Substring(0, [Math]::Min(60, $out.Length))
        if ($out -match "not recognized|not found|error|Cannot find") {
            Write-Host "MISS $($c.name.PadRight(10)) $short" -ForegroundColor Red
        } else {
            Write-Host "OK   $($c.name.PadRight(10)) $short" -ForegroundColor Green
        }
    } catch {
        Write-Host "MISS $($c.name)" -ForegroundColor Red
    }
}
