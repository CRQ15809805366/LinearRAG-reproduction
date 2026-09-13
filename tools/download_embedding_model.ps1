param(
    [string]$Destination = "model/all-MiniLM-L6-v2"
)

$ErrorActionPreference = "Stop"
$revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
$baseUrl = "https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/resolve/$revision"
$files = @(
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "model.safetensors",
    "1_Pooling/config.json"
)

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Destination "1_Pooling") | Out-Null

foreach ($file in $files) {
    $target = Join-Path $Destination $file
    Write-Host "Downloading $file"
    & curl.exe -fL --retry 5 --retry-all-errors --connect-timeout 20 "$baseUrl/$file" -o $target
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed for $file with exit code $LASTEXITCODE"
    }
}

$expectedModelSha256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
$actualModelSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $Destination "model.safetensors")).Hash.ToLowerInvariant()
if ($actualModelSha256 -ne $expectedModelSha256) {
    throw "Model SHA-256 mismatch: $actualModelSha256"
}

Write-Host "Model ready at $Destination (revision $revision)."
