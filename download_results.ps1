$runs = @(
    @{dist="iid"; mode="noisy";     pauli="0.038"},
    @{dist="iid";     mode="noisy";     pauli="0.045"},
    @{dist="iid";     mode="noisy";     pauli="0.07"},
    @{dist="iid";     mode="mitigated"; pauli="0.038"},
    @{dist="iid";     mode="mitigated"; pauli="0.045"},
    @{dist="iid";     mode="mitigated"; pauli="0.07"}
)

foreach ($r in $runs) {
    $sub = "scaled_p$($r.pauli)_r$($r.pauli)_s0.002"
    $remote = "formenti@montblanc.mi.infn.it:~/qibo-qfl-torch/results/$($r.dist)/fedavg/$($r.mode)/$sub/nshots_1000"
    $localDest = "C:/Users/fede4/qibo-qfl/qibo-qfl-torch/results/$($r.dist)/fedavg/$($r.mode)/$sub"
    
    New-Item -ItemType Directory -Force -Path $localDest | Out-Null
    Write-Host ">>> $($r.dist) $($r.mode) p=$($r.pauli)" -ForegroundColor Cyan
    scp -r -J federico.formenti1@tolab.fisica.unimi.it $remote $localDest
}