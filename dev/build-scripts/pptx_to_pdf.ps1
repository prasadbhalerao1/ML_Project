param([string]$In, [string]$Out)
$ErrorActionPreference = 'Stop'
$app = New-Object -ComObject PowerPoint.Application
try {
    # ReadOnly=false is required for SaveCopyAs, but the source is untouched:
    # SaveCopyAs writes a new file and leaves the open presentation unmodified.
    $pres = $app.Presentations.Open($In, $false, $false, $false)
    $pres.SaveCopyAs($Out, 32)   # 32 = ppSaveAsPDF
    $pres.Close()
    "PDF written"
} finally { $app.Quit(); [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null }
