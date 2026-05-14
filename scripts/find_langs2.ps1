# Deep search for ruby, node, kotlin, crystal, julia
$targets = @("ruby.exe","node.exe","kotlinc.bat","kotlinc","crystal.exe","julia.exe")
$searchRoots = @("C:\","C:\ProgramData","C:\Users\hinge\AppData\Local","C:\Users\hinge\AppData\Roaming")

foreach ($root in $searchRoots) {
    if (-not (Test-Path $root)) { continue }
    Get-ChildItem $root -Recurse -ErrorAction SilentlyContinue -Depth 6 |
        Where-Object { $targets -contains $_.Name } |
        Select-Object FullName |
        ForEach-Object { Write-Host "FOUND: $($_.FullName)" }
}
