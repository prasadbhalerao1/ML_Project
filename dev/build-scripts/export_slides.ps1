param([string]$In, [string]$OutDir, [int[]]$Slides, [int]$Width = 1600)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force $OutDir | Out-Null
$app = New-Object -ComObject PowerPoint.Application
try {
    # ReadOnly, Untitled, WithWindow=false -> never touches the source file.
    $pres = $app.Presentations.Open($In, -1, 0, 0)
    $h = [int]($Width * $pres.PageSetup.SlideHeight / $pres.PageSetup.SlideWidth)
    foreach ($n in $Slides) {
        $pres.Slides($n).Export((Join-Path $OutDir ("slide{0:D2}.png" -f $n)), "PNG", $Width, $h)
    }
    $pres.Close()
    "exported: " + ($Slides -join ",")
} finally { $app.Quit(); [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null }
