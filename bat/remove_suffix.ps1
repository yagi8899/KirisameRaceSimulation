Param(
    [string]$Dir = "models",
    [string]$Suffix = "_2015-2024",
    [string]$Ext = ".sav",
    [switch]$WhatIf
)

try {
    Write-Output "Models dir: $Dir"
    Write-Output "Suffix: $Suffix"
    Write-Output "Ext: $Ext"

    $pattern = "*" + $Suffix + $Ext
    $files = Get-ChildItem -Path $Dir -Filter $pattern -File -ErrorAction SilentlyContinue

    if (-not $files -or $files.Count -eq 0) {
        Write-Output "No files found."
        exit 0
    }

    $count = 0
    foreach ($f in $files) {
        $newName = $f.Name -replace [regex]::Escape($Suffix), ''
        if ($WhatIf) {
            Write-Output ("{0} -> {1} (whatif)" -f $f.Name, $newName)
        }
        else {
            Rename-Item -Path $f.FullName -NewName $newName -ErrorAction Stop
            Write-Output ("{0} -> {1}" -f $f.Name, $newName)
        }
        $count++
    }

    Write-Output "`n$count files renamed"
    exit 0
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}
