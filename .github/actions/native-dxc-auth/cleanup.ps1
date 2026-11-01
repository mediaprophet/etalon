param([string]$Directory = $env:DXC_CURL_HOME)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($Directory)) { return }

$parent = [IO.Path]::GetFullPath($env:RUNNER_TEMP).TrimEnd([IO.Path]::DirectorySeparatorChar)
$target = [IO.Path]::GetFullPath($Directory).TrimEnd([IO.Path]::DirectorySeparatorChar)
$leaf = [IO.Path]::GetFileName($target)
if ([IO.Path]::GetDirectoryName($target) -ne $parent -or $leaf -notmatch '^etalon-static-dxc-[a-f0-9]{32}$') {
    throw 'Refusing cleanup outside the owned runner-temp directory.'
}
if (-not (Test-Path -LiteralPath $target)) { return }
$item = Get-Item -LiteralPath $target
if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    throw 'Refusing cleanup of a redirected or non-directory resource.'
}
$marker = Join-Path $target 'owner.txt'
if (-not (Test-Path -LiteralPath $marker -PathType Leaf) -or [IO.File]::ReadAllText($marker) -cne $leaf) {
    throw 'Refusing cleanup without the expected ownership marker.'
}
# Delete only our ordinary files; never recursively traverse caller content.
$allowed = @('owner.txt', '.curlrc', 'credentials.netrc', 'release.json', 'headers.txt')
$entries = @(Get-ChildItem -LiteralPath $target -Force)
foreach ($entry in $entries) {
    if ($entry.PSIsContainer -or ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $entry.Name -cnotin $allowed) {
        throw 'Unexpected content in the owned credential directory.'
    }
}
foreach ($entry in $entries) { Remove-Item -LiteralPath $entry.FullName -Force }
Remove-Item -LiteralPath $target -Force
