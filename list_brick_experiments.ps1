param(
    [string]$BricksRoot = "D:/datasets/dataset_v2/KITTI_DEYOLO_v2/bricks"
)

function Count-Png($dir) {
    if (Test-Path $dir) {
        return (Get-ChildItem $dir -Filter *.png -File -ErrorAction SilentlyContinue | Measure-Object).Count
    } else {
        return 0
    }
}

if (-not (Test-Path $BricksRoot)) {
    Write-Host "❌ Bricks root not found: $BricksRoot"
    exit 1
}

Write-Host "🔍 Listing brick experiment folders under $BricksRoot`n"

Get-ChildItem -Path $BricksRoot -Directory | ForEach-Object {
    $expPath = $_.FullName
    $expName = $_.Name
    Write-Host "=== $expName ==="

    # modes: invd, log, invd_denoised, mask, hag, grad, range_strip, etc.
    Get-ChildItem -Path $expPath -Directory | ForEach-Object {
        $modeName = $_.Name
        $train = Join-Path $_.FullName "ir_train"
        $val   = Join-Path $_.FullName "ir_val"
        $test  = Join-Path $_.FullName "ir_test"

        $ctTrain = Count-Png $train
        $ctVal   = Count-Png $val
        $ctTest  = Count-Png $test

        "{0,-15} -> train:{1,5}  val:{2,5}  test:{3,5}" -f $modeName, $ctTrain, $ctVal, $ctTest
    }
    Write-Host ""
}
