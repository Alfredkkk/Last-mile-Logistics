$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath '.').Path
$archiveRoot = [System.IO.Path]::GetFullPath((Join-Path $taskRoot 'archive\validation_2026-09-21'))
$rootPrefix = $taskRoot.TrimEnd('\') + '\'
$archivePrefix = $archiveRoot.TrimEnd('\') + '\'
if (-not $archiveRoot.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Archive target is outside the project'
}
$mapping = @(
    @{ Source = 'tests'; Target = 'tests' },
    @{ Source = 'tmp\review_2026-09-05'; Target = 'review_2026-09-05' },
    @{ Source = 'tmp\fixes_2026-09-21'; Target = 'fixes' }
)
$records = @()
# Verify every absolute source/destination before any directory move.
foreach ($item in $mapping) {
    $source = (Resolve-Path -LiteralPath (Join-Path $taskRoot $item.Source)).Path
    $destination = [System.IO.Path]::GetFullPath((Join-Path $archiveRoot $item.Target))
    if (-not $source.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
        -not $destination.StartsWith($archivePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Move would leave the intended project/archive directory'
    }
    if (Test-Path -LiteralPath $destination) { throw "Archive destination already exists: $destination" }
    $files = @(Get-ChildItem -LiteralPath $source -Recurse -File -Force)
    $checks = @()
    foreach ($file in $files) {
        $relative = $file.FullName.Substring($source.Length + 1)
        if ($relative -notmatch '(^|[\\/])(\.venv|__pycache__)([\\/]|$)') {
            $checks += @{ relative_path = $relative; size = $file.Length;
                         sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash }
        }
    }
    $records += @{ source = $item.Source; destination = ('archive\validation_2026-09-21\' + $item.Target);
                   absolute_source = $source; absolute_destination = $destination;
                   file_count = $files.Count; total_bytes = [long](($files | Measure-Object -Property Length -Sum).Sum);
                   checks = $checks; verified = $false }
}
New-Item -ItemType Directory -Path $archiveRoot -Force | Out-Null
foreach ($record in $records) {
    Move-Item -LiteralPath $record.absolute_source -Destination $record.absolute_destination
    $files = @(Get-ChildItem -LiteralPath $record.absolute_destination -Recurse -File -Force)
    $bytes = [long](($files | Measure-Object -Property Length -Sum).Sum)
    if ($files.Count -ne $record.file_count -or $bytes -ne $record.total_bytes) {
        throw "Archive file count/size mismatch for $($record.source)"
    }
    foreach ($check in $record.checks) {
        $actual = (Get-FileHash -LiteralPath (Join-Path $record.absolute_destination $check.relative_path) -Algorithm SHA256).Hash
        if ($actual -ne $check.sha256) { throw "Archive hash mismatch: $($check.relative_path)" }
    }
    $record.verified = $true
    Write-Output "Archived and verified $($record.source): $($record.file_count) files"
}
$oldLogs = @(Get-ChildItem -LiteralPath (Join-Path $taskRoot 'archive\training_logs') -Recurse -File -Filter '*.csv' |
    ForEach-Object { @{ path = $_.FullName.Substring($taskRoot.Length + 1);
                       sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash } })
$manifest = @{ archived_after_tests = 53; mappings = $records; historical_logs_already_archived = $oldLogs;
               hash_scope = 'Source/evidence files; dependency environment and bytecode checked by file count and total bytes' }
$json = $manifest | ConvertTo-Json -Depth 12
[System.IO.File]::WriteAllText((Join-Path $archiveRoot 'archive_manifest.json'), $json, [System.Text.UTF8Encoding]::new($false))
