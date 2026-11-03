# Convert a "# %%"-cell Python script into a Jupyter notebook for Kaggle's File > Import Notebook.
# Usage: powershell -File tools\py2ipynb.ps1 notebooks\01_eda.py   (writes notebooks\01_eda.ipynb)
param([Parameter(Mandatory)][string]$Path)

$Path = (Resolve-Path $Path).Path
$src = [IO.File]::ReadAllText($Path)
$chunks = [regex]::Split($src, "(?m)^# %%") | Select-Object -Skip 1
$cells = @()
foreach ($c in $chunks) {
  $isMd = $c.StartsWith(" [markdown]")
  $i = $c.IndexOf("`n")
  $body = if ($i -ge 0) { $c.Substring($i + 1).Trim("`r", "`n") } else { "" }
  $lines = $body -split "`r?`n"
  if ($isMd) { $lines = $lines | ForEach-Object { if ($_.StartsWith("# ")) { $_.Substring(2) } else { $_.TrimStart("#") } } }
  $n = @($lines).Count
  $srcLines = for ($k = 0; $k -lt $n; $k++) { if ($k -lt $n - 1) { $lines[$k] + "`n" } else { $lines[$k] } }
  $cell = [ordered]@{ cell_type = $(if ($isMd) { "markdown" } else { "code" }); metadata = @{}; source = @($srcLines) }
  if (-not $isMd) { $cell.execution_count = $null; $cell.outputs = @() }
  $cells += $cell
}
$nb = [ordered]@{
  cells = $cells
  metadata = @{ kernelspec = @{ display_name = "Python 3"; language = "python"; name = "python3" }; language_info = @{ name = "python" } }
  nbformat = 4; nbformat_minor = 5
}
$out = [IO.Path]::ChangeExtension($Path, ".ipynb")
[IO.File]::WriteAllText($out, ($nb | ConvertTo-Json -Depth 10), (New-Object Text.UTF8Encoding $false))
"$out ($($cells.Count) cells)"
