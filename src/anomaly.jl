# ══════════════════════════════════════════════════════════════════════════════
#  anomaly.jl — anomaly detection in emissions
#
#  Anomaly detection in a carbon inventory has three jobs, in this order:
#
#   1. find emission events that the plant did not intend (leaks, flare upset,
#      uncombusted slip, a controlled variable drifting);
#   2. find *data* problems before they become reporting problems (a meter stuck
#      at a constant value, a factor applied to the wrong unit);
#   3. alert without crying wolf — the false-alarm rate has to be a design
#      parameter, not an accident.
#
#  Five complementary detectors are implemented, from simple and interpretable to
#  multivariate and non-linear:
#
#     r̂ₜ = yₜ − ŷₜ            residual from the soft sensor (or any model)
#     zₜ = λ r̂ₜ + (1−λ)z_{t−1}         EWMA chart          (slow drifts)
#     Sₜ⁺/Sₜ⁻ = max(0, S + r̂ₜ ∓ k)     CUSUM chart         (small persistent shifts)
#     T² = (x⃗−μ⃗)ᵀ Σ⁻¹ (x⃗−μ⃗)          Hotelling T²        (multivariate outliers)
#     SPE = ‖x⃗ − P Pᵀ x⃗‖²             PCA Q-statistic     (breaks in correlation)
#     SPE_AE = ‖x⃗ − f_θ⃗(x⃗)‖²         autoencoder         (non-linear structure)
#
#  Every threshold is calibrated on the *training* period to a target false-alarm
#  rate, so ARL₀ = 1/α samples between false alarms is a documented property of
#  the system rather than a hope.
# ══════════════════════════════════════════════════════════════════════════════

"""
    AnomalyEvent

One flagged observation: when, how strongly, by which detector, what the value
was, the limit it crossed, and how confident the attribution is.
"""
struct AnomalyEvent
    t::Int
    method::String
    score::Float64
    limit::Float64
    severity::Symbol        # :low | :medium | :high
    feature::String
    value::Float64
    note::String
end

function Base.show(io::IO, e::AnomalyEvent)
    @printf(io, "t=%-5d %-16s score=%9.3f > limit=%9.3f  [%s] %s", e.t, e.method, e.score,
            e.limit, uppercase(string(e.severity)), e.note)
end

"Severity of an exceedance from the ratio score/limit."
function severity_of(score::Real, limit::Real)
    limit <= 0 && return :low
    r = score / limit
    r >= 2.0 && return :high
    r >= 1.3 && return :medium
    :low
end

"""
    ewma_chart(r̂; λ=0.2, L=3.0, σ̂=nothing) -> NamedTuple

Exponentially weighted moving average chart:

    zₜ = λ r̂ₜ + (1 − λ) z_{t−1},        σ_z = σ̂ √(λ/(2 − λ)),        limits ±L σ_z

The EWMA is the detector for *slow* drifts (a seal degrading over weeks) which a
Shewhart chart only sees after they are large.
"""
function ewma_chart(r̂::AbstractVector{<:Real}; λ::Real=0.2, L::Real=3.0,
                    σ̂::Union{Nothing,Real}=nothing)
    σ = σ̂ === nothing ? std(Float64.(r̂)) : Float64(σ̂)
    σ = σ > 0 ? σ : 1.0
    z = zeros(length(r̂))
    zz = 0.0
    for t in eachindex(r̂)
        zz = λ * r̂[t] + (1 - λ) * zz
        z[t] = zz
    end
    σ_z = σ * sqrt(λ / (2 - λ))
    limit = L * σ_z
    (z=z, limit=limit, σ_z=σ_z, λ=λ, L=L,
     flags=[abs(zi) > limit for zi in z], σ̂=σ)
end

"""
    cusum_chart(r̂; k=0.5, h=5.0, σ̂=nothing) -> NamedTuple

Two-sided CUSUM with reference value `k` (in units of σ̂) and decision interval `h`:

    Sₜ⁺ = max(0, S_{t−1}⁺ + r̂ₜ − k σ̂)
    Sₜ⁻ = min(0, S_{t−1}⁻ + r̂ₜ + k σ̂)

Optimal for detecting a persistent shift of 1σ–2σ — the signature of a leaking
flange or an abatement device that stopped working.
"""
function cusum_chart(r̂::AbstractVector{<:Real}; k::Real=0.5, h::Real=5.0,
                     σ̂::Union{Nothing,Real}=nothing)
    σ = σ̂ === nothing ? std(Float64.(r̂)) : Float64(σ̂)
    σ = σ > 0 ? σ : 1.0
    Sp, Sm = zeros(length(r̂)), zeros(length(r̂))
    sp, sm = 0.0, 0.0
    for t in eachindex(r̂)
        sp = max(0.0, sp + r̂[t] - k * σ)
        sm = min(0.0, sm + r̂[t] + k * σ)
        Sp[t], Sm[t] = sp, sm
    end
    (S_plus=Sp, S_minus=Sm, limit=h * σ, k=k * σ, σ̂=σ,
     flags=[(Sp[t] > h * σ) || (Sm[t] < -h * σ) for t in eachindex(r̂)])
end

"""
    calibrate_threshold(scores; α=0.01) -> NamedTuple

Non-parametric control limit for a target false-alarm rate α: the empirical
`1 − α` quantile of the scores observed while the process was under control. No
distributional assumption, because emission-residual distributions are heavy
tailed more often than they are Gaussian.
"""
function calibrate_threshold(scores::AbstractVector{<:Real}; α::Real=0.01)
    s = sort(Float64.(scores))
    isempty(s) && return (limit=Inf, α=α, ARL₀=Inf, n=0)
    idx = clamp(ceil(Int, (1 - α) * length(s)), 1, length(s))
    (limit=s[idx], α=α, ARL₀=1 / α, n=length(s))
end

"Expected number of samples between false alarms, ARL₀ = 1/α."
ARL0(α::Real) = 1 / α

# ── multivariate detectors ───────────────────────────────────────────────────
"""
    fit_covariance(X; ridge=1e-8) -> (μ⃗, Σ)

Mean and (regularised) covariance of the reference period. The ridge term keeps Σ
invertible when tags are collinear — which, in a plant historian, they often are.
"""
function fit_covariance(X::AbstractMatrix{<:Real}; ridge::Real=1e-8)
    Xf = Float64.(X)
    p = size(Xf, 2)
    (vec(mean(Xf; dims=1)), cov(Xf) + ridge * Matrix{Float64}(I, p, p))
end

"""
    hotelling_t2(X; reference, μ, Σ, α=0.05) -> NamedTuple

Hotelling statistic `T² = (x⃗ − μ⃗)ᵀ Σ⁻¹ (x⃗ − μ⃗)` of every observation against the
(supposedly normal) reference period. The limit is the empirical `1 − α` quantile
of the reference statistics, so the false-alarm rate is exact by construction
rather than assumed.
"""
function hotelling_t2(X::AbstractMatrix{<:Real};
                      reference::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                      μ::Union{Nothing,AbstractVector{<:Real}}=nothing,
                      Σ::Union{Nothing,AbstractMatrix{<:Real}}=nothing, α::Real=0.05)
    ref = reference === nothing ? X : reference
    p = size(X, 2)
    μv = μ === nothing ? first(fit_covariance(ref)) : Float64.(μ)
    Σm = Σ === nothing ? last(fit_covariance(ref)) : Float64.(Σ)
    Σinv = inv(Σm)
    score(A) = [dot(x - μv, Σinv * (x - μv)) for x in eachrow(Float64.(A))]
    T² = score(X)
    lim = quantile(score(ref), 1 - α)
    (T²=T², limit=lim, μ=μv, Σ=Σm, flags=[t > lim for t in T²], df=p)
end

"""
    pca_spe(X; k=2, reference, α=0.01) -> NamedTuple

Principal-component monitoring, the classical MSPC pair:

    T²  = Σ_{i≤k} (tᵢ/sᵢ)²            variation *inside* the normal subspace
    SPE = Q = ‖x⃗ − P Pᵀ x⃗‖²         variation *outside* it (squared prediction error)

A degrading seal that changes the *correlation* between flow and pressure shows up
in SPE long before any single tag looks unusual — which is why both statistics are
reported.
"""
function pca_spe(X::AbstractMatrix{<:Real}; k::Integer=2,
                 reference::Union{Nothing,AbstractMatrix{<:Real}}=nothing, α::Real=0.01)
    ref = Float64.(reference === nothing ? X : reference)
    μ = vec(mean(ref; dims=1))
    R = ref .- μ'
    F = svd(R)
    kk = clamp(Int(k), 1, min(size(R)...))
    P = F.V[:, 1:kk]
    s = F.S[1:kk] ./ sqrt(max(size(R, 1) - 1, 1))
    score_T²(A) = begin
        T = (Float64.(A) .- μ') * P
        [sum((T[i, :] ./ max.(s, eps())) .^ 2) for i in 1:size(T, 1)]
    end
    score_SPE(A) = begin
        B = Float64.(A) .- μ'
        [sum((B[i, :] - P * (P' * B[i, :])) .^ 2) for i in 1:size(B, 1)]
    end
    evr = F.S .^ 2 ./ sum(F.S .^ 2)
    (T²=score_T²(X), SPE=score_SPE(X),
     T²_limit=quantile(score_T²(ref), 1 - α), SPE_limit=quantile(score_SPE(ref), 1 - α),
     P=P, μ=μ, k=kk, explained_variance=sum(evr[1:kk]), α=α)
end

# ── autoencoder: non-linear reconstruction error ─────────────────────────────
"""
    Autoencoder

A bottleneck network trained to reproduce its own input,

    x⃗ → a⁽¹⁾ → … → a⁽ᵏ⁾ (bottleneck) → … → x̂ ≈ x⃗,

trained with the same explicit back-propagation and Adam step as the soft sensor.
The score of an observation is its reconstruction error

    SPE_AE = ‖x⃗ − f_θ⃗(x⃗)‖²

small for anything resembling normal operation — including non-linear correlations
that PCA cannot see — and large for genuinely new behaviour.
"""
struct Autoencoder
    W::Vector{Matrix{Float64}}
    b::Vector{Vector{Float64}}
    acts::Vector{Symbol}
    xμ::Vector{Float64}
    xσ::Vector{Float64}
    history::Vector{NamedTuple{(:epoch, :train, :val),Tuple{Int,Float64,Float64}}}
    bottleneck::Int
end

function Base.show(io::IO, ae::Autoencoder)
    print(io, "Autoencoder(", length(ae.W[1]), "→", join([size(w, 1) for w in ae.W], "→"),
          ", bottleneck ", ae.bottleneck, ", ", length(ae.history), " epochs)")
end

"""
    train_autoencoder(X; hidden=[8], epochs, η, batch, λ_L2, seed) -> Autoencoder

Train the bottleneck network on the normal-operation period, with early stopping on
a held-out split of that period: the reconstruction error must stay calibrated, so
the network may not learn to reproduce anomalies it has never seen.
"""
function train_autoencoder(X::AbstractMatrix{<:Real}; hidden::AbstractVector{<:Integer}=[8],
                           acts::Symbol=:tanh, epochs::Integer=600, η::Real=0.01,
                           batch::Integer=32, λ_L2::Real=1e-4, val_fraction::Real=0.25,
                           patience::Integer=80, seed::Integer=1)
    Xf = Float64.(X)
    n, p = size(Xf)
    Xs, xμ, xσ = standardize!(Xf)
    net = MLP(p, vcat(hidden, [p]); n_out=p, acts=acts, seed=seed, loss="mse")

    rng = MersenneTwister(seed)
    perm = randperm(rng, n)
    nval = clamp(round(Int, val_fraction * n), 1, max(1, n - 2))
    val_idx, tr_idx = perm[1:nval], perm[(nval + 1):end]

    mW = [zeros(size(w)) for w in net.W]
    vW = [zeros(size(w)) for w in net.W]
    mb = [zeros(size(b)) for b in net.b]
    vb = [zeros(size(b)) for b in net.b]
    hist = NamedTuple{(:epoch, :train, :val),Tuple{Int,Float64,Float64}}[]
    best = (loss=Inf, W=[copy(w) for w in net.W], b=[copy(b) for b in net.b], epoch=0)
    left = Int(patience)
    t = 0

    function sample_grad(i)
        a⃗s, z⃗s, x̂ = forward(net, Xs[i, :])
        δ = x̂ .- Xs[i, :]                       # ∂(½‖x̂ − x‖²)/∂x̂
        gW = [zeros(size(w)) for w in net.W]
        gb = [zeros(size(b)) for b in net.b]
        gW[end] = δ * transpose(a⃗s[end - 1]) .+ λ_L2 .* net.W[end]
        gb[end] = copy(δ)
        for l in (length(net.W) - 1):-1:1
            δ = (transpose(net.W[l + 1]) * δ) .* dact(net.acts[l], z⃗s[l])
            gW[l] = δ * transpose(a⃗s[l]) .+ λ_L2 .* net.W[l]
            gb[l] = copy(δ)
        end
        (0.5 * sum(abs2, x̂ .- Xs[i, :]), gW, gb)
    end

    for epoch in 1:epochs
        order = randperm(rng, length(tr_idx))
        for s in 1:batch:length(order)
            idxs = tr_idx[order[s:min(end, s + batch - 1)]]
            gW = [zeros(size(w)) for w in net.W]
            gb = [zeros(size(b)) for b in net.b]
            for i in idxs
                _, dW, db = sample_grad(i)
                for l in eachindex(gW)
                    gW[l] .+= dW[l]
                    gb[l] .+= db[l]
                end
            end
            m = length(idxs)
            t += 1
            for l in eachindex(net.W)
                adam_step!(net.W[l], gW[l] ./ m, mW[l], vW[l], t, η, 0.9, 0.999, 1e-8)
                adam_step!(net.b[l], gb[l] ./ m, mb[l], vb[l], t, η, 0.9, 0.999, 1e-8)
            end
        end
        trL = mean(sample_grad(i)[1] for i in tr_idx)
        vaL = mean(sample_grad(i)[1] for i in val_idx)
        push!(hist, (epoch=epoch, train=trL, val=vaL))
        if vaL < best.loss - 1e-12
            best = (loss=vaL, W=[copy(w) for w in net.W], b=[copy(b) for b in net.b], epoch=epoch)
            left = Int(patience)
        else
            left -= 1
            left <= 0 && break
        end
    end
    for l in eachindex(net.W)
        copyto!(net.W[l], best.W[l])
        copyto!(net.b[l], best.b[l])
    end
    Autoencoder(net.W, net.b, net.acts, xμ, xσ, hist,
                isempty(hidden) ? p : hidden[end])
end

"The autoencoder's weights wrapped as an `MLP` view (reuses `forward`/`predict`)."
_ae_net(ae::Autoencoder) = MLP(ae.W, ae.b, ae.acts, ae.xμ, ae.xσ, 0.0, 1.0,
                               size(ae.W[end], 1), "mse", length(ae.history), [])

"Reconstructed input in the original units of the tags."
function reconstruct(ae::Autoencoder, X::AbstractMatrix{<:Real})
    Xs = (Float64.(X) .- ae.xμ') ./ ae.xσ'
    permutedims(reduce(hcat, [forward(_ae_net(ae), r)[3] for r in eachrow(Xs)]))
end

"""
    reconstruction_error(ae, X; per_feature=false) -> Vector or Matrix

`SPE_AE = ‖x⃗ − x̂‖²` per observation (the anomaly score), or the matrix of squared
per-feature errors when `per_feature = true` — which is what allows attribution:
*which tag* makes this hour look strange.
"""
function reconstruction_error(ae::Autoencoder, X::AbstractMatrix{<:Real}; per_feature::Bool=false)
    Xs = (Float64.(X) .- ae.xμ') ./ ae.xσ'
    net = _ae_net(ae)
    E = Matrix{Float64}(undef, size(X, 1), size(X, 2))
    for (i, r) in enumerate(eachrow(Xs))
        x̂ = forward(net, r)[3]
        E[i, :] = (x̂ .- r) .^ 2
    end
    per_feature ? E : vec(sum(E; dims=2))
end

"""
    feature_attribution(ae, X, names) -> DataFrame

Ranked share of the reconstruction error per tag — the first answer to *why* the
model considers an hour anomalous.
"""
function feature_attribution(ae::Autoencoder, X::AbstractMatrix{<:Real},
                             names::AbstractVector{<:AbstractString})
    E = reconstruction_error(ae, X; per_feature=true)
    s = vec(sum(E; dims=1))
    tot = sum(s)
    df = DataFrame(feature=String.(names), squared_error=s,
                   share=[tot > 0 ? si / tot : 0.0 for si in s])
    sort(df, :share, rev=true)
end

# ── composition: detectors → events → case report ────────────────────────────
"""
    residual_frame(observed, predicted; t) -> DataFrame

Residual table `r̂ₜ = yₜ − ŷₜ` with its standardised form — the input of every
residual detector.
"""
function residual_frame(observed::AbstractVector{<:Real}, predicted::AbstractVector{<:Real};
                        t::Union{Nothing,AbstractVector}=nothing)
    y = Float64.(observed)
    ŷ = Float64.(predicted)
    r = y .- ŷ
    σ = std(r)
    σ = σ > 0 ? σ : 1.0
    DataFrame(t=t === nothing ? collect(1:length(r)) : Float64.(t), observed=y, predicted=ŷ,
              r̂=r, r̂_z=r ./ σ, σ̂_r=σ)
end

"""
    AnomalyReport

Everything the detector suite produced: the events, a per-detector summary and the
false-alarm rate the limits were calibrated to.
"""
struct AnomalyReport
    events::Vector{AnomalyEvent}
    summary::DataFrame
    α::Real
    detectors::Vector{String}
end

function Base.show(io::IO, r::AnomalyReport)
    print(io, "AnomalyReport(", length(r.events), " event(s) from ",
          join(r.detectors, " + "), ", target false-alarm rate α = ", r.α, ")")
end

"Count events per detector and severity, with a rate per 1000 observations."
function summarize_events(events::AbstractVector{AnomalyEvent}, n_obs::Int)
    dets = sort(unique([e.method for e in events]))
    isempty(dets) && return DataFrame(detector=String[], severity=String[], n=Int[], rate_per_1000=Float64[])
    rows = NamedTuple{(:detector, :severity, :n),Tuple{String,String,Int}}[]
    for d in dets, s in ["low", "medium", "high"]
        n = count(e -> e.method == d && string(e.severity) == s, events)
        n > 0 && push!(rows, (detector=d, severity=s, n=n))
    end
    df = DataFrame(rows)
    df.rate_per_1000 = 1000 .* df.n ./ max(n_obs, 1)
    df
end

"Events as a DataFrame, for the case-management table of the notebook."
events_dataframe(events::AbstractVector{AnomalyEvent}) = DataFrame(
    t=[e.t for e in events], detector=[e.method for e in events],
    score=[e.score for e in events], limit=[e.limit for e in events],
    severity=[string(e.severity) for e in events], feature=[e.feature for e in events],
    value=[e.value for e in events], note=[e.note for e in events])

"""
    detect_anomalies(series::AbstractVector; α, λ, L, k, h, feature, t0) -> AnomalyReport

Univariate residual suite — EWMA (drift), CUSUM (persistent shift) and a calibrated
single-point threshold — applied to `r̂ₜ`. Every limit derives from the series' own
in-control spread, and `α` (default 1 %) fixes ARL₀ = 100 samples between false
alarms.
"""
function detect_anomalies(series::AbstractVector{<:Real}; α::Real=0.01, λ::Real=0.2,
                          L::Real=3.0, k::Real=0.5, h::Real=5.0,
                          feature::AbstractString="residual", t0::Integer=1)
    r = Float64.(series)
    σ̂ = std(r)
    e = ewma_chart(r; λ=λ, L=L, σ̂=σ̂)
    c = cusum_chart(r; k=k, h=h, σ̂=σ̂)
    cal = calibrate_threshold(abs.(r); α=α)
    events = AnomalyEvent[]
    for t in eachindex(r)
        e.flags[t] && push!(events, AnomalyEvent(t0 + t - 1, "EWMA", abs(e.z[t]), e.limit,
            severity_of(abs(e.z[t]), e.limit), String(feature), r[t],
            "smoothed residual outside ±$(L)σ_z (λ = $λ) — slow drift"))
        if c.flags[t]
            s = max(abs(c.S_plus[t]), abs(c.S_minus[t]))
            push!(events, AnomalyEvent(t0 + t - 1, "CUSUM", s, c.limit,
                severity_of(s, c.limit), String(feature), r[t],
                "cumulative shift beyond $(h)σ̂ (k = $k) — persistent change"))
        end
        abs(r[t]) > cal.limit && push!(events, AnomalyEvent(t0 + t - 1, "threshold", abs(r[t]),
            cal.limit, severity_of(abs(r[t]), cal.limit), String(feature), r[t],
            "single-point residual beyond the 1−α quantile (ARL₀ = $(round(cal.ARL₀; digits=0)))"))
    end
    AnomalyReport(events, summarize_events(events, length(r)), α, ["EWMA", "CUSUM", "threshold"])
end

"""
    detect_anomalies(X::AbstractMatrix; reference, α, k_pca, ae_hidden, names, seed) -> AnomalyReport

Multivariate suite on the continuous tag matrix — Hotelling T² (inside the normal
subspace), PCA SPE/Q (outside it) and autoencoder reconstruction error (non-linear),
all limits taken from the `reference` period at the same confidence level.
"""
function detect_anomalies(X::AbstractMatrix{<:Real};
                          reference::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                          α::Real=0.05, k_pca::Integer=2,
                          ae_hidden::AbstractVector{<:Integer}=[8],
                          names::Union{Nothing,AbstractVector}=nothing,
                          ae_epochs::Integer=400, seed::Integer=1)
    ref = reference === nothing ? X : reference
    p = size(X, 2)
    names_v = names === nothing ? ["x$i" for i in 1:p] : String.(names)
    h = hotelling_t2(X; reference=ref, α=α)
    pc = pca_spe(X; k=k_pca, reference=ref, α=α)
    ae = train_autoencoder(ref; hidden=ae_hidden, epochs=ae_epochs, seed=seed)
    spe_ae = reconstruction_error(ae, X)
    lim_ae = quantile(reconstruction_error(ae, ref), 1 - α)
    top = feature_attribution(ae, X, names_v).feature[1]

    events = AnomalyEvent[]
    for t in 1:size(X, 1)
        h.flags[t] && push!(events, AnomalyEvent(t, "Hotelling T²", h.T²[t], h.limit,
            severity_of(h.T²[t], h.limit), "joint tags", h.T²[t],
            "combination of tags outside the normal cloud"))
        pc.SPE[t] > pc.SPE_limit && push!(events, AnomalyEvent(t, "PCA SPE", pc.SPE[t],
            pc.SPE_limit, severity_of(pc.SPE[t], pc.SPE_limit), "correlation break", pc.SPE[t],
            "correlation between tags broken (Q-statistic)"))
        spe_ae[t] > lim_ae && push!(events, AnomalyEvent(t, "Autoencoder SPE", spe_ae[t], lim_ae,
            severity_of(spe_ae[t], lim_ae), top, spe_ae[t],
            "non-linear reconstruction error; dominant tag: $top"))
    end
    AnomalyReport(events, summarize_events(events, size(X, 1)), α,
                  ["Hotelling T²", "PCA SPE", "Autoencoder SPE"])
end

"""
    anomaly_emission_impact(flags, E_t; baseline) -> NamedTuple

Translate detections into emissions: how much CO₂e was emitted during flagged
intervals and how much of it is *excess* over the expected level. This is the number
that decides whether an alert becomes a corrective action or a footnote.
"""
function anomaly_emission_impact(flags::AbstractVector{Bool}, E_t::AbstractVector{<:Real};
                                 baseline::Union{Nothing,AbstractVector{<:Real}}=nothing)
    E = Float64.(E_t)
    n = min(length(flags), length(E))
    base = baseline === nothing ? fill(mean(E), n) : Float64.(baseline)[1:n]
    f = flags[1:n]
    E_flag = sum(E[1:n][f]; init=0.0)
    E_excess = sum(max.(E[1:n][f] .- base[f], 0.0); init=0.0)
    (E_flagged=E_flag, E_excess=E_excess, share_flagged=E_flag / max(sum(E[1:n]), eps()),
     n_flagged=count(f))
end

"""
    log_anomalies!(ledger, actor, report; context) -> Int

Write the medium- and high-severity detections to the audit trail as *proposed
investigation* entries, so that "we saw it and we investigated it" is provable at
the next audit.
"""
function log_anomalies!(ledger::AuditLedger, actor::Actor, report::AnomalyReport;
                        context::AbstractString="anomaly detection run")
    n = 0
    for e in report.events
        e.severity == :low && continue
        record!(ledger, actor, "propose", "anomaly@t=$(e.t)";
                why="@$(e.method): $(e.note) — $context",
                after=@sprintf("score %.3f > limit %.3f (%s)", e.score, e.limit, e.severity))
        n += 1
    end
    n
end

"""
    anomaly_report(rep::AnomalyReport; per_hour, ledger, actor) -> NamedTuple

Final packaging of a detection run: the case table, the per-detector summary, the
emission impact (when a per-hour emission table is supplied) and the audit-trail
entries opened for the medium/high findings.
"""
function anomaly_report(rep::AnomalyReport; per_hour::Union{Nothing,DataFrame}=nothing,
                        ledger::Union{Nothing,AuditLedger}=nothing,
                        actor::Union{Nothing,Actor}=nothing, fdr::Real=0.01)
    impact = nothing
    if per_hour !== nothing && nrow(per_hour) > 0
        flagged = falses(nrow(per_hour))
        E = per_hour.E_t
        base = fill(mean(E), nrow(per_hour))
        for e in rep.events
            e.severity == :low && continue
            (1 <= e.t <= nrow(per_hour)) && (flagged[e.t] = true)
        end
        impact = anomaly_emission_impact(flagged, E; baseline=base)
    end
    logged = (ledger !== nothing && actor !== nothing) ?
             log_anomalies!(ledger, actor, rep; context="detection run, FDR target $fdr") : 0
    (cases=events_dataframe(rep.events), summary=rep.summary, impact=impact, logged=logged,
     detectors=rep.detectors, α=rep.α)
end
