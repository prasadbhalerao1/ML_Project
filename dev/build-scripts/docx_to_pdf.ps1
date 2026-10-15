param([string]$In, [string]$Out)
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($In, [ref]$false, [ref]$true)
    # Refresh fields so the page-number footer renders.
    $doc.Fields.Update() | Out-Null
    $doc.SaveAs([ref]$Out, [ref]17)   # 17 = wdFormatPDF
    $doc.Close([ref]$false)
    Write-Output "PDF written"
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
