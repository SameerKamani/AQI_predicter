Param(
  [int]$Days = 1095,
  [int]$MaterializeDays = 90,
  [switch]$ImputeShortGaps,
  [int]$MinHoursPerDay = 20
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-Step {
  param(
    [string]$Message,
    [scriptblock]$Action
  )
  Write-Host "`n=== $Message ===" -ForegroundColor Cyan
  & $Action
}

# Move to repo root (directory of this script)
Set-Location -Path $PSScriptRoot

Invoke-Step -Message "Install/verify Python dependencies" -Action {
  python -m pip install -r requirements.txt
}

Invoke-Step -Message "Build offline features (last $Days days, excluding today)" -Action {
  $env:PYTHONUNBUFFERED = "1"   # ensure real-time Python output
  $impute = ""
  if ($ImputeShortGaps) { $impute = "--impute_short_gaps" }
  python -u Data_Collection/feature_store_pipeline.py --days $Days $impute --min_hours_per_day $MinHoursPerDay
}

Invoke-Step -Message "Register Feast objects" -Action {
  Push-Location feature_repo
  feast apply
  Pop-Location
}

Invoke-Step -Message "Materialize features to Feast online store" -Action {
  $start = ((Get-Date).ToUniversalTime().AddDays(-1 * $MaterializeDays)).ToString('s') + 'Z'
  $end   = (Get-Date).ToUniversalTime().ToString('s') + 'Z'
  Push-Location feature_repo
  feast materialize $start $end
  Pop-Location
}

Invoke-Step -Message "Verify online vs offline consistency" -Action {
  Push-Location feature_repo
  python -u .\verify_online_offline_consistency.py
  Pop-Location
}

Invoke-Step -Message "Run EDA on offline features" -Action {
  python -u EDA/run_eda.py
}

Write-Host "`nAll steps completed successfully." -ForegroundColor Green

