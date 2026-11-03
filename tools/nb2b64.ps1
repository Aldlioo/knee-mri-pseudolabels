# Convert a "# %%" script to .ipynb (py2ipynb.ps1), gzip + base64 it, write the text to -Out and print it.
# Usage: powershell -File tools\nb2b64.ps1 notebooks\09_label_lab.py C:\path\to\out_b64.txt
param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][string]$Out)
$Path = (Resolve-Path $Path).Path
& "$PSScriptRoot\py2ipynb.ps1" $Path | Out-Null
$nb = [IO.Path]::ChangeExtension($Path, ".ipynb")
$bytes = [IO.File]::ReadAllBytes($nb)
$ms = New-Object IO.MemoryStream
$gz = New-Object IO.Compression.GZipStream($ms, [IO.Compression.CompressionMode]::Compress)
$gz.Write($bytes, 0, $bytes.Length); $gz.Close()
$b64 = [Convert]::ToBase64String($ms.ToArray())
[IO.File]::WriteAllText($Out, $b64)
$b64
