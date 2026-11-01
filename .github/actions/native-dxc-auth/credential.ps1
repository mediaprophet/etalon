function ConvertTo-DxcNetrc {
    param([AllowNull()][AllowEmptyString()][string]$Token)

    if ([string]::IsNullOrEmpty($Token) -or $Token.Length -gt 8192) {
        throw 'Credential is missing or exceeds the 8192-byte limit.'
    }
    foreach ($character in $Token.ToCharArray()) {
        if ([char]::IsControl($character) -or [int]$character -in @(0x2028, 0x2029)) {
            throw 'Credential contains a forbidden control or line separator.'
        }
    }
    try { $bytes = [Text.UTF8Encoding]::new($false, $true).GetByteCount($Token) }
    catch { throw 'Credential is not valid Unicode.' }
    if ($bytes -gt 8192) { throw 'Credential exceeds the 8192-byte limit.' }

    # curl 7.84+ netrc quoted strings preserve spaces, quotes and escaped backslashes.
    $quoted = '"' + $Token.Replace('\', '\\').Replace('"', '\"') + '"'
    return "machine api.github.com`nlogin x-access-token`npassword $quoted`n"
}
