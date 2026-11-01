Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

if (-not $IsWindows) { throw 'This transport is scoped to the Windows Core MSRV job.' }
. (Join-Path $PSScriptRoot 'credential.ps1')
$token = $env:DXC_GITHUB_TOKEN
$netrcContent = ConvertTo-DxcNetrc -Token $token
if ([string]::IsNullOrWhiteSpace($env:RUNNER_TEMP) -or [string]::IsNullOrWhiteSpace($env:GITHUB_OUTPUT)) {
    throw 'The GitHub runner temporary directory and step output are required.'
}

$leaf = 'etalon-static-dxc-' + [Guid]::NewGuid().ToString('N')
$directory = Join-Path ([IO.Path]::GetFullPath($env:RUNNER_TEMP)) $leaf
$previousCurlHome = $env:CURL_HOME
New-Item -ItemType Directory -Path $directory | Out-Null
[IO.File]::WriteAllText((Join-Path $directory 'owner.txt'), $leaf)
# Publish the owned path before creating credentials so cancellation cleanup can find it.
[IO.File]::AppendAllText($env:GITHUB_OUTPUT, "curl-home=$directory`n", [Text.UTF8Encoding]::new($false))
try {
    # Keep credential files private to the runner account, including inherited children.
    $acl = Get-Acl -LiteralPath $directory
    $acl.SetAccessRuleProtection($true, $false)
    $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User
    $rule = [Security.AccessControl.FileSystemAccessRule]::new(
        $sid, 'FullControl', 'ContainerInherit, ObjectInherit', 'None', 'Allow')
    $acl.AddAccessRule($rule)
    Set-Acl -LiteralPath $directory -AclObject $acl

    $netrc = Join-Path $directory 'credentials.netrc'
    [IO.File]::WriteAllText($netrc, $netrcContent, [Text.UTF8Encoding]::new($false))
    $netrcPath = $netrc.Replace('\', '/')
    $config = @"
fail
silent
show-error
connect-timeout = 20
max-time = 180
retry = 3
retry-max-time = 300
max-redirs = 5
proto = "=https"
proto-redir = "=https"
netrc-optional
netrc-file = "$netrcPath"
"@
    [IO.File]::WriteAllText((Join-Path $directory '.curlrc'), $config, [Text.UTF8Encoding]::new($false))
    $env:CURL_HOME = $directory
    $headers = Join-Path $directory 'headers.txt'
    $body = Join-Path $directory 'release.json'
    $url = 'https://api.github.com/repos/DouglasDwyer/mach-dxcompiler/releases/tags/2026.10.26%2B141dd7f.1'
    $status = & curl.exe --location --max-filesize 2097152 --dump-header $headers --output $body --write-out '%{http_code}' $url
    $curlStatus = $LASTEXITCODE
    $headerLines = if (Test-Path -LiteralPath $headers) { @(Get-Content -LiteralPath $headers) } else { @() }
    foreach ($line in $headerLines) {
        if ($line -match '^(HTTP/\S+\s+\d{3}|x-ratelimit-(limit|remaining|reset|resource):|retry-after:|x-github-request-id:)') {
            Write-Host $line
        }
    }
    if ($curlStatus -ne 0 -or "$status" -cne '200') {
        throw "Static DXC metadata request failed (curl=$curlStatus, HTTP=$status); immutability was not determined."
    }
    $limitLine = @($headerLines | Where-Object { $_ -match '^x-ratelimit-limit:\s*\d+\s*$' })
    if ($limitLine.Count -eq 0 -or [long](($limitLine[-1] -split ':', 2)[1].Trim()) -le 60) {
        throw 'The metadata request did not establish an authenticated GitHub rate-limit bucket.'
    }
    if ((Get-Item -LiteralPath $body).Length -gt 2097152) { throw 'Release metadata exceeded its byte budget.' }
    $release = [IO.File]::ReadAllText($body) | ConvertFrom-Json -Depth 30
    if (($release.id -isnot [long] -and $release.id -isnot [int]) -or
        $release.tag_name -isnot [string] -or $release.id -ne 424477231 -or $release.tag_name -cne '2026.10.26+141dd7f.1' -or
        $release.immutable -isnot [bool] -or -not $release.immutable -or
        $release.draft -isnot [bool] -or $release.draft -or $release.assets -isnot [array]) {
        throw 'Release identity, schema or immutable status did not match the pinned declaration.'
    }
    $expected = @(
        @{ Name = 'x86_64-windows-msvc_ReleaseFast_lib.tar.gz'; Size = 26261723; Id = 678755984; Hash = '908305e46a7f100b240f29235b0844006ced9c0cf22c95d019e3d59c14346725' },
        @{ Name = 'x86_64-windows-msvc_ReleaseFast_Dynamic_lib.tar.gz'; Size = 26115520; Id = 678750221; Hash = 'abe009a4abb88dab9a1659ca12515e35baa34478a2ed42103595ebf65f8f6b25' }
    )
    foreach ($item in $expected) {
        $assets = @($release.assets | Where-Object { $_.name -ceq $item.Name })
        if ($assets.Count -ne 1) { throw 'The pinned native archive is missing or duplicated.' }
        $asset = $assets[0]
        $download = 'https://github.com/DouglasDwyer/mach-dxcompiler/releases/download/2026.10.26%2B141dd7f.1/' + $item.Name
        if (($asset.id -isnot [long] -and $asset.id -isnot [int]) -or
            ($asset.size -isnot [long] -and $asset.size -isnot [int]) -or $asset.id -ne $item.Id -or $asset.size -ne $item.Size -or $asset.state -cne 'uploaded' -or
            $asset.digest -cne ('sha256:' + $item.Hash) -or $asset.browser_download_url -cne $download) {
            throw 'Native archive identity, length, digest or URL did not match the pinned declaration.'
        }
        Write-Host "Verified immutable release metadata: $($item.Name), $($item.Size) bytes, SHA-256 $($item.Hash)"
    }
} catch {
    & (Join-Path $PSScriptRoot 'cleanup.ps1') -Directory $directory
    throw
} finally {
    $env:CURL_HOME = $previousCurlHome
    $token = $null
    $netrcContent = $null
}
