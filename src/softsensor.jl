# ══════════════════════════════════════════════════════════════════════════════
#  softsensor.jl — a neural soft sensor for the annual analyzer campaign
#
#  The operational reality this module addresses: the expensive measurement
#  (a mobile analyzer, a lab campaign, a certified stack test) happens **once a
#  year for one week** — N_camp = 7 d · 24 h = 168 samples per source — while the
#  plant produces continuous data every hour. A soft sensor learns the mapping
#
#       ŷ = f_θ⃗(x⃗)          x⃗ : continuous DCS/SCADA tags (temperature, pressure,
#                                  flow, O₂, load, feedstock, hours since service)
#                            y : the variable the campaign measures
#                                (CH₄ mole fraction, flare DRE, N₂O rate, VOC …)
#
#  and then predicts `y` all year, which turns 168 measurements into 8 760
#  emission estimates with a *quantified* uncertainty instead of a smoothed
#  annual average.
#
#  The network is implemented from scratch — explicit forward pass and
#  back-propagation, Adam optimiser — precisely so that the equations above can
#  be checked line by line:
#
#       z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
#       a⁽ˡ⁾ = g(z⁽ˡ⁾)
#       δ⁽ᴸ⁾ = ∂L/∂ŷ ,  δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ δ⁽ˡ⁺¹⁾ ⊙ g′(z⁽ˡ⁾)
#       ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ (a⁽ˡ⁻¹⁾)ᵀ + λ_L2 W⁽ˡ⁾
# ══════════════════════════════════════════════════════════════════════════════

"Activation `g(·)` of a hidden layer."
function act(kind::Symbol, z::AbstractVector{<:Real})
    kind == :tanh && return tanh.(z)
    kind == :relu && return max.(z, 0.0)
    kind == :softplus && return log1p.(exp.(clamp.(z, -50.0, 50.0)))
    throw(ArgumentError("unknown activation «$kind»"))
end

"Derivative `g′(z)`, expressed through the pre-activation z (and a for tanh)."
function dact(kind::Symbol, z::AbstractVector{<:Real})
    kind == :tanh && return 1 .- tanh.(z) .^ 2
    kind == :relu && return Float64.(z .> 0)
    kind == :softplus && return 1 ./ (1 .+ exp.(-clamp.(z, -50.0, 50.0)))
    throw(ArgumentError("unknown activation «$kind»"))
end

"""
    MLP

A fully connected soft sensor. Weights are stored per layer as `W⁽ˡ⁾` (matrix) and
`b⁽ˡ⁾` (vector); features and target are standardised, which is what keeps the
training well conditioned on tags that differ by six orders of magnitude
(a pressure in bar next to a flow in m³/h).

`n_out = 1` → point prediction; `n_out = 2` → `[μ, log σ̂²]`, i.e. the network also
learns its own noise level (heteroscedastic regression), which is the honest way
to report a soft-sensor uncertainty.
"""
struct MLP
    W::Vector{Matrix{Float64}}
    b::Vector{Vector{Float64}}
    acts::Vector{Symbol}
    xμ::Vector{Float64}
    xσ::Vector{Float64}
    yμ::Float64
    yσ::Float64
    n_out::Int
    loss::String          # "mse" | "nll"
    trained_epochs::Int
    history::Vector{NamedTuple{(:epoch, :train, :val),Tuple{Int,Float64,Float64}}}
end

"Number of parameters θ⃗ of the network."
nparams(net::MLP) = sum(length, net.W) + sum(length, net.b)

"One-line summary of the architecture."
function Base.show(io::IO, net::MLP)
    arch = join([size(w, 1) for w in net.W], "×")
    print(io, "MLP($(net.n_out == 2 ? "heteroscedastic" : "point"), layers ", size(net.W[1], 2), "→",
          arch, ", ", nparams(net), " θ, ", net.loss, ", ", net.trained_epochs, " epochs)")
end

"Glorot-uniform initialisation of W⁽ˡ⁾ and zero biases."
function _xavier!(rng, n_in::Int, n_out::Int)
    limit = sqrt(6.0 / (n_in + n_out))
    (rand(rng, n_out, n_in) .* 2 .- 1) .* limit
end

"""
    MLP(n_in, hidden; n_out=1, acts=:tanh, seed=1, loss=nothing)

Construct an untrained soft sensor with `hidden = [16, 8]`-style layer sizes.
"""
function MLP(n_in::Int, hidden::AbstractVector{<:Integer}; n_out::Int=1,
             acts::Symbol=:tanh, seed::Integer=1, loss::Union{Nothing,String}=nothing)
    rng = MersenneTwister(seed)
    sizes = vcat(n_in, hidden, n_out)
    W = [_xavier!(rng, sizes[i], sizes[i + 1]) for i in 1:length(sizes) - 1]
    b = [zeros(sizes[i + 1]) for i in 1:length(sizes) - 1]
    a = fill(acts, length(hidden))
    l = loss === nothing ? (n_out == 2 ? "nll" : "mse") : loss
    MLP(W, b, a, zeros(n_in), ones(n_in), 0.0, 1.0, n_out, l, 0,
        NamedTuple{(:epoch, :train, :val),Tuple{Int,Float64,Float64}}[])
end

"Standardise a feature matrix with the training statistics (xᵢ ← (xᵢ − μᵢ)/σᵢ)."
function standardize!(X::AbstractMatrix{<:Real})
    μ = vec(mean(X; dims=1))
    σ = vec(std(X; dims=1))
    σ = map(s -> s < 1e-12 ? 1.0 : s, σ)
    ((X .- μ') ./ σ', μ, σ)
end

"Apply stored standardisation statistics to new features."
apply_standardisation(net::MLP, X::AbstractMatrix{<:Real}) = (X .- net.xμ') ./ net.xσ'

"""
    forward(net, x⃗) -> (a⃗s, z⃗s, ŷ)

Forward pass for one sample, keeping every intermediate `a⁽ˡ⁾` and `z⁽ˡ⁾` because
back-propagation needs them:

    z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾,   a⁽ˡ⁾ = g(z⁽ˡ⁾),   ŷ = W⁽ᴸ⁾a⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾
"""
function forward(net::MLP, x⃗::AbstractVector{<:Real})
    a⃗s = Vector{Vector{Float64}}(undef, length(net.W) + 1)
    z⃗s = Vector{Vector{Float64}}(undef, length(net.W))
    a⃗s[1] = collect(Float64, x⃗)
    for l in eachindex(net.W)
        z⃗s[l] = net.W[l] * a⃗s[l] + net.b[l]
        a⃗s[l + 1] = l <= length(net.acts) ? act(net.acts[l], z⃗s[l]) : z⃗s[l]
    end
    (a⃗s, z⃗s, a⃗s[end])
end

"""
    loss_and_grad(net, x⃗, y_std; λ_L2) -> (loss, gradW, gradb, ŷ)

Loss and its exact gradients for one sample (target already standardised):

  * `loss = "mse"` — `L = ½(ŷ − y)²`, `δ⁽ᴸ⁾ = ŷ − y`
  * `loss = "nll"` — heteroscedastic Gaussian NLL with `s = log σ̂²`:
    `L = ½[(μ − y)²e^{−s} + s]`,  `∂L/∂μ = (μ − y)e^{−s}`,  `∂L/∂s = ½[1 − (μ − y)²e^{−s}]`

Hidden-layer error signals follow the chain rule

    δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ δ⁽ˡ⁺¹⁾ ⊙ g′(z⁽ˡ⁾),      ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ + λ_L2 W⁽ˡ⁾
"""
function loss_and_grad(net::MLP, x⃗::AbstractVector{<:Real}, y_std::Real; λ_L2::Real=0.0)
    a⃗s, z⃗s, ŷ = forward(net, x⃗)
    L, δ = if net.loss == "nll"
        μ, s = ŷ[1], clamp(ŷ[2], -8.0, 8.0)
        r = μ - y_std
        e = exp(-s)
        (0.5 * (r^2 * e + s), [r * e, 0.5 * (1 - r^2 * e)])
    else
        r = ŷ[1] - y_std
        (0.5 * r^2, [r])
    end
    gradW = [zeros(size(w)) for w in net.W]
    gradb = [zeros(size(b)) for b in net.b]
    gradW[end] = δ * transpose(a⃗s[end - 1]) .+ λ_L2 .* net.W[end]
    gradb[end] = copy(δ)
    for l in (length(net.W) - 1):-1:1
        δ = (transpose(net.W[l + 1]) * δ) .* dact(net.acts[l], z⃗s[l])
        gradW[l] = δ * transpose(a⃗s[l]) .+ λ_L2 .* net.W[l]
        gradb[l] = copy(δ)
    end
    (L, gradW, gradb, ŷ)
end

"Adam update of one parameter array: θ ← θ − η·m̂/(√v̂ + ε)."
function adam_step!(θ::AbstractArray, g::AbstractArray, m::AbstractArray, v::AbstractArray,
                    t::Integer, η::Real, β₁::Real, β₂::Real, ε::Real)
    @. m = β₁ * m + (1 - β₁) * g
    @. v = β₂ * v + (1 - β₂) * g^2
    m̂ = @. m / (1 - β₁^t)
    v̂ = @. v / (1 - β₂^t)
    @. θ -= η * m̂ / (sqrt(v̂) + ε)
    θ
end

"Raw network output for one standardised feature vector."
predict_std(net::MLP, x_std::AbstractVector{<:Real}) = forward(net, x_std)[3]

"""
    predict(net, X) -> Vector{Float64}

Point prediction `ŷ = f_θ⃗(x⃗)` in the original units of the campaign measurement:
features are standardised with the stored statistics and the network output is
mapped back with `ŷ = yμ + yσ · f(x⃗_std)`.
"""
predict(net::MLP, X::AbstractMatrix{<:Real}) =
    [net.yμ + net.yσ * forward(net, collect(Float64, r))[3][1]
     for r in eachrow(apply_standardisation(net, X))]

predict(net::MLP, x⃗::AbstractVector{<:Real}) = predict(net, permutedims(collect(Float64, x⃗)))

"""
    predict_interval(net, X; z=1.96) -> Vector{NamedTuple}

Predictive distribution of a heteroscedastic soft sensor: the mean comes from the
first output, the standard deviation from the second
(`σ̂ = yσ · exp(½·log σ̂²_std)`), giving the `ŷ ± z·σ̂` interval.
"""
function predict_interval(net::MLP, X::AbstractMatrix{<:Real}; z::Real=1.96)
    out = NamedTuple{(:ŷ, :σ̂, :lo, :hi),Tuple{Float64,Float64,Float64,Float64}}[]
    for r in eachrow(apply_standardisation(net, X))
        y = forward(net, collect(Float64, r))[3]
        ŷ = net.yμ + net.yσ * y[1]
        σ̂ = net.n_out == 2 ? net.yσ * exp(0.5 * clamp(y[2], -8.0, 8.0)) : NaN
        push!(out, (ŷ=ŷ, σ̂=σ̂, lo=ŷ - z * σ̂, hi=ŷ + z * σ̂))
    end
    out
end

"Rebuild a network carrying training statistics, epoch count and loss history."
function _with_stats(net::MLP, xμ, xσ, yμ, yσ, epochs::Int, hist)
    MLP(net.W, net.b, net.acts, xμ, xσ, yμ, yσ, net.n_out, net.loss, epochs, hist)
end

"Mean loss over a set of samples (regularisation excluded, so runs stay comparable)."
_epoch_loss(net::MLP, Xs, ys, idx; λ_L2::Real=0.0) =
    mean(loss_and_grad(net, Xs[i, :], ys[i]; λ_L2=λ_L2)[1] for i in idx)

"""
    train_soft_sensor(X, y; hidden, epochs, η, batch, λ_L2, val_fraction, patience, seed, heteroscedastic)

Train the soft sensor with mini-batch Adam and early stopping on a validation split:

    θ ← θ − η · m̂/(√v̂ + ε)        (Adam, β₁ = 0.9, β₂ = 0.999)
    L(θ⃗) = MSE or Gaussian NLL, regularised by λ_L2‖W‖²

Returns an `MLP` whose `history` field carries the train/validation curve of every
epoch — the evidence that the model is not over-fitted. With
`heteroscedastic = true` the network has two outputs and learns `ŷ` **and**
`σ̂(x⃗)`, which is what makes a per-hour uncertainty statement possible.
"""
function train_soft_sensor(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
                           hidden::AbstractVector{<:Integer}=[16, 8], acts::Symbol=:tanh,
                           heteroscedastic::Bool=false, η::Real=0.01, epochs::Integer=600,
                           batch::Integer=32, λ_L2::Real=1e-4, β₁::Real=0.9, β₂::Real=0.999,
                           ε::Real=1e-8, val_fraction::Real=0.25, patience::Integer=80,
                           seed::Integer=1)
    n, p = size(X)
    n >= 6 || throw(ArgumentError("a soft sensor needs at least a handful of campaign samples"))
    rng = MersenneTwister(seed)
    Xs, xμ, xσ = standardize!(Float64.(X))
    yμ, yσ = mean(y), std(y)
    yσ = yσ < 1e-12 ? 1.0 : yσ
    ys = (Float64.(y) .- yμ) ./ yσ
    loss = heteroscedastic ? "nll" : "mse"
    net = MLP(p, hidden; n_out=(heteroscedastic ? 2 : 1), acts=acts, seed=seed, loss=loss)

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
    for epoch in 1:epochs
        order = randperm(rng, length(tr_idx))
        for s in 1:batch:length(order)
            idxs = tr_idx[order[s:min(end, s + batch - 1)]]
            gW = [zeros(size(w)) for w in net.W]
            gb = [zeros(size(b)) for b in net.b]
            for i in idxs
                _, dW, db, _ = loss_and_grad(net, Xs[i, :], ys[i]; λ_L2=λ_L2)
                for l in eachindex(gW)
                    gW[l] .+= dW[l]
                    gb[l] .+= db[l]
                end
            end
            m = length(idxs)
            t += 1
            for l in eachindex(net.W)
                adam_step!(net.W[l], gW[l] ./ m, mW[l], vW[l], t, η, β₁, β₂, ε)
                adam_step!(net.b[l], gb[l] ./ m, mb[l], vb[l], t, η, β₁, β₂, ε)
            end
        end
        trL = _epoch_loss(net, Xs, ys, tr_idx)
        vaL = _epoch_loss(net, Xs, ys, val_idx)
        push!(hist, (epoch=epoch, train=trL, val=vaL))
        if vaL < best.loss - 1e-10
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
    _with_stats(net, xμ, xσ, yμ, yσ, best.epoch, hist)
end

"Training history as a DataFrame (epoch, train, val) — the convergence evidence."
training_history(net::MLP) = DataFrame(epoch=[h.epoch for h in net.history],
                                       train=[h.train for h in net.history],
                                       val=[h.val for h in net.history])

"""
    SoftSensorMetrics

Quality of a soft sensor: `RMSE`, `MAE`, `R²`, mean bias and — when predictive
intervals exist — the empirical 95 % coverage. A coverage of 0.95 means the stated
uncertainty is honest; 0.70 means the model is over-confident, which in a
compliance context matters more than a slightly worse RMSE.
"""
struct SoftSensorMetrics
    n::Int
    RMSE::Float64
    MAE::Float64
    R²::Float64
    bias::Float64
    coverage95::Float64
    σ̂_mean::Float64
end

function Base.show(io::IO, m::SoftSensorMetrics)
    @printf(io, "n=%d  RMSE=%.4f  MAE=%.4f  R²=%.4f  bias=%+.4f  coverage95=%.1f%%  σ̂=%.4f",
            m.n, m.RMSE, m.MAE, m.R², m.bias, 100 * m.coverage95, m.σ̂_mean)
end

"""
    softsensor_metrics(y, ŷ; intervals) -> SoftSensorMetrics

Regression scores plus the empirical coverage of the predictive interval when the
sensor is heteroscedastic: `coverage95 = P[y ∈ ŷ ± 1.96 σ̂]`.
"""
function softsensor_metrics(y::AbstractVector{<:Real}, ŷ::AbstractVector{<:Real};
                            intervals::Union{Nothing,Vector}=nothing)
    y = Float64.(y)
    ŷ = Float64.(ŷ)
    r = ŷ .- y
    sst = sum((y .- mean(y)) .^ 2)
    R² = sst > 0 ? 1 - sum(r .^ 2) / sst : NaN
    cov, σ̂m = 0.0, 0.0
    if intervals !== nothing
        inside = [intervals[i].lo <= y[i] <= intervals[i].hi for i in eachindex(y)]
        cov = count(inside) / length(y)
        σ̂m = mean([intervals[i].σ̂ for i in eachindex(y)])
    end
    SoftSensorMetrics(length(y), sqrt(mean(r .^ 2)), mean(abs.(r)), R², mean(r), cov, σ̂m)
end

"Metrics as a one-row DataFrame, for reports."
metrics_table(m::SoftSensorMetrics) = DataFrame(n=[m.n], RMSE=[m.RMSE], MAE=[m.MAE],
    R²=[m.R²], bias=[m.bias], coverage95=[m.coverage95], σ̂_mean=[m.σ̂_mean])

"""
    train_ensemble(X, y; n_models=5, seeds=[1,…,n], kwargs...) -> Vector{MLP}

Bagged ensemble: every member sees the same campaign but a different initialisation
and (through the internal validation split) a different data ordering. The spread
between members is the **epistemic** uncertainty — how much the answer depends on
which week happened to be measured.
"""
function train_ensemble(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
                        n_models::Integer=5, seeds::AbstractVector{<:Integer}=collect(1:5),
                        kwargs...)
    [train_soft_sensor(X, y; seed=s, kwargs...) for s in seeds[1:min(n_models, length(seeds))]]
end

"""
    ensemble_interval(nets, X; z=1.96) -> Vector{NamedTuple}

Combined predictive interval, the number to carry into the inventory uncertainty
budget:

    σ_total² = σ_epistemic² + σ̂_aleatoric²
    σ_epistemic = std over ensemble members,   σ̂_aleatoric = mean heteroscedastic σ̂
"""
function ensemble_interval(nets::AbstractVector{MLP}, X::AbstractMatrix{<:Real}; z::Real=1.96)
    per_member = [predict_interval(net, X; z=z) for net in nets]
    n = size(X, 1)
    out = NamedTuple{(:ŷ, :σ_epi, :σ_ale, :σ_total, :lo, :hi),
                     Tuple{Float64,Float64,Float64,Float64,Float64,Float64}}[]
    for i in 1:n
        ŷs = [pm[i].ŷ for pm in per_member]
        ale = [pm[i].σ̂ for pm in per_member if !isnan(pm[i].σ̂)]
        σ_ale = isempty(ale) ? 0.0 : mean(ale)
        ŷm = mean(ŷs)
        σ_epi = length(ŷs) > 1 ? std(ŷs) : 0.0
        σ_tot = sqrt(σ_epi^2 + σ_ale^2)
        push!(out, (ŷ=ŷm, σ_epi=σ_epi, σ_ale=σ_ale, σ_total=σ_tot, lo=ŷm - z * σ_tot, hi=ŷm + z * σ_tot))
    end
    out
end

"""
    cross_validate_softsensor(X, y; k=5, kwargs...) -> DataFrame

k-fold cross-validation *of the campaign*: each fold holds out a block of campaign
hours, so the reported error estimates how the sensor behaves on an operating
condition it has not seen.
"""
function cross_validate_softsensor(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
                                   k::Integer=5, seed::Integer=1, kwargs...)
    n = size(X, 1)
    k = clamp(k, 2, max(2, n ÷ 3))
    perm = randperm(MersenneTwister(seed), n)
    folds = [perm[i:k:n] for i in 1:k]
    rows = NamedTuple{(:fold, :n_train, :n_test, :RMSE, :MAE, :R²),
                      Tuple{Int,Int,Int,Float64,Float64,Float64}}[]
    for (j, test_idx) in enumerate(folds)
        train_idx = setdiff(1:n, test_idx)
        net = train_soft_sensor(X[train_idx, :], y[train_idx]; seed=seed + j, kwargs...)
        ŷ = predict(net, X[test_idx, :])
        m = softsensor_metrics(y[test_idx], ŷ)
        push!(rows, (fold=j, n_train=length(train_idx), n_test=length(test_idx),
                     RMSE=m.RMSE, MAE=m.MAE, R²=m.R²))
    end
    DataFrame(fold=[r.fold for r in rows], n_train=[r.n_train for r in rows],
              n_test=[r.n_test for r in rows], RMSE=[r.RMSE for r in rows],
              MAE=[r.MAE for r in rows], R²=[r.R² for r in rows])
end

# ── campaign design: does one week represent the year? ───────────────────────
"Inverse standard-normal CDF (Acklam's rational approximation, |ε| < 1.15e-9)."
function quantile_normal(p::Real)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow
        q = sqrt(-2 * log(p))
        return (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
               ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
    elseif p <= phigh
        q = p - 0.5
        r = q * q
        return (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q /
               (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1)
    end
    q = sqrt(-2 * log(1 - p))
    -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
     ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
end

"χ² quantile by Wilson–Hilferty (≈1 % accurate for df ≥ 2); the limit of T² and D²."
function chi2_quantile(p::Real, df::Integer)
    z = quantile_normal(p)
    df * (1 - 2 / (9df) + z * sqrt(2 / (9df)))^3
end

"""
    min_campaign_samples(σ̂; δ, z=1.96) -> Int

Sample-size rule for the annual campaign: to estimate the campaign mean with half
width δ at confidence level `z`, `N ≥ (z σ̂ / δ)²`. It puts the analyser budget
where it changes the answer rather than where it is convenient.
"""
min_campaign_samples(σ̂::Real; δ::Real, z::Real=1.96) = ceil(Int, (z * σ̂ / δ)^2)

"""
    campaign_design(X_campaign, X_annual; feature_names, activity, α=0.05) -> NamedTuple

Quantify whether the one-week campaign represents the whole year. Two independent
tests are applied to the continuous tag matrix:

  1. **envelope** — per feature, the fraction of annual hours whose value lies
     inside the campaign's [min, max];
  2. **Mahalanobis** — `D² = (x⃗ − μ_c)ᵀ Σ_c⁻¹ (x⃗ − μ_c)` of every annual sample
     against the campaign cloud, compared with the χ² limit `χ²_{1−α,p}`. This
     catches combinations of tags that are individually plausible but jointly
     unseen — the dangerous kind of extrapolation.

`risk_share` weights the uncovered hours by `activity` (flare volume, production),
i.e. it answers: *what share of the emissions is predicted outside the envelope?*
"""
function campaign_design(X_campaign::AbstractMatrix{<:Real}, X_annual::AbstractMatrix{<:Real};
                         feature_names::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
                         activity::Union{Nothing,AbstractVector{<:Real}}=nothing, α::Real=0.05)
    p = size(X_campaign, 2)
    names = feature_names === nothing ? ["x$i" for i in 1:p] : String.(feature_names)
    rows = NamedTuple{(:feature, :campaign_min, :campaign_max, :annual_min, :annual_max,
                       :covered_fraction),Tuple{String,Float64,Float64,Float64,Float64,Float64}}[]
    for j in 1:p
        cmin, cmax = extrema(X_campaign[:, j])
        amin, amax = extrema(X_annual[:, j])
        inside = count(x -> cmin <= x <= cmax, X_annual[:, j])
        push!(rows, (feature=names[j], campaign_min=cmin, campaign_max=cmax,
                     annual_min=amin, annual_max=amax,
                     covered_fraction=inside / size(X_annual, 1)))
    end
    μ = vec(mean(X_campaign; dims=1))
    Σ = cov(Float64.(X_campaign)) + 1e-8 * Matrix{Float64}(I, p, p)
    Σinv = inv(Σ)
    D² = [dot(x - μ, Σinv * (x - μ)) for x in eachrow(Float64.(X_annual))]
    lim = chi2_quantile(1 - α, p)
    coverage = count(d -> d <= lim, D²) / length(D²)
    outside = [d > lim for d in D²]
    risk_share = activity === nothing ? count(outside) / length(outside) :
                 sum(Float64.(activity)[outside]) / max(sum(Float64.(activity)), eps())
    worst = sort(collect(rows), by=r -> r.covered_fraction)[1]
    recommendation = if coverage >= 0.9 && risk_share <= 0.10
        "campaign envelope is representative (coverage $(round(100*coverage; digits=1)) %, " *
        "emission-weighted risk $(round(100*risk_share; digits=1)) %)"
    elseif coverage >= 0.8
        "acceptable, but the weakest tag is «$(worst.feature)» " *
        "(only $(round(100*worst.covered_fraction; digits=1)) % of annual hours inside the campaign range): " *
        "stratify the next campaign by that variable"
    else
        "the one-week campaign does NOT represent the year " *
        "(coverage $(round(100*coverage; digits=1)) %, emission-weighted risk $(round(100*risk_share; digits=1)) %): " *
        "extend or stratify the campaign across load states, or carry a wider uncertainty " *
        "on the extrapolated hours"
    end
    (features=DataFrame(rows), coverage=coverage, chi2_limit=lim, risk_share=risk_share,
     D²=D², recommendation=recommendation)
end

# ── annualisation: 168 campaign hours → 8 760 annual estimates ───────────────
"""
    annualize_with_softsensor(nets, X_annual, activity; GWP₁₀₀, hours, z, target_scale)

Propagate the soft sensor over the whole year:

    m_t = Q̇_t · (ŷ_t · target_scale)      (gas mass per interval, e.g. kg/h)
    E   = GWP₁₀₀ · Σₜ m_t · Δt / 1000     (tCO₂e per year)

`target_scale` converts the predicted target into a mass per unit of activity —
e.g. for a prediction of a *volume percentage* of methane, `target_scale = ρ_CH₄/100`
turns volume-% into kg of CH₄ per m³. Being explicit here is what keeps a soft
sensor from silently over- or under-stating a year by two orders of magnitude.

Returns the annual total, its interval (ensemble epistemic + heteroscedastic
aleatoric spread), the per-hour table — which the anomaly engine consumes next —
and the relative uncertainty to hand to the inventory's uncertainty budget.
"""
function annualize_with_softsensor(nets::AbstractVector{MLP}, X_annual::AbstractMatrix{<:Real},
                                   activity::AbstractVector{<:Real};
                                   GWP₁₀₀::Real=GWP_AR6.CH₄_fossil, hours::Real=1.0,
                                   z::Real=1.96, target_scale::Real=1.0)
    iv = ensemble_interval(nets, X_annual; z=z)
    a = Float64.(activity) .* target_scale
    m = a .* [i.ŷ for i in iv] .* hours
    m_lo = a .* [i.lo for i in iv] .* hours
    m_hi = a .* [i.hi for i in iv] .* hours
    E = GWP₁₀₀ * sum(m) / 1000
    lo = GWP₁₀₀ * sum(m_lo) / 1000
    hi = GWP₁₀₀ * sum(m_hi) / 1000
    per_hour = DataFrame(t=collect(1:length(m)), activity=Float64.(activity),
        ŷ=[i.ŷ for i in iv], σ_epi=[i.σ_epi for i in iv], σ_ale=[i.σ_ale for i in iv],
        lo=[i.lo for i in iv], hi=[i.hi for i in iv], m=m, E_t=GWP₁₀₀ .* m ./ 1000)
    (E=E, lo=lo, hi=hi, u_rel=E > 0 ? (hi - lo) / (2z * E) : 0.0, per_hour=per_hour,
     GWP₁₀₀=GWP₁₀₀, target_scale=target_scale,
     method="soft sensor over annual continuous data")
end

annualize_with_softsensor(net::MLP, X::AbstractMatrix{<:Real}, activity::AbstractVector{<:Real}; kwargs...) =
    annualize_with_softsensor([net], X, activity; kwargs...)

"""
    annualize_ratio(activity_annual, campaign_activity, campaign_emission) -> NamedTuple

The *incumbent* method, for comparison in the notebook: scale the campaign-week
emission by the ratio of annual to campaign activity,

    E_year = E_campaign · (Q̇_year / Q̇_campaign)

It is still what most inventories do; it implicitly assumes the emission factor
does not depend on load, temperature or feedstock — the assumption the soft sensor
tests rather than assumes.
"""
function annualize_ratio(activity_annual::AbstractVector{<:Real},
                         campaign_activity::AbstractVector{<:Real}, campaign_emission::Real)
    ratio = sum(Float64.(activity_annual)) / max(sum(Float64.(campaign_activity)), eps())
    (E=campaign_emission * ratio, ratio=ratio, method="activity-ratio scaling of the campaign week")
end

"Disclosure sentence for the annualised figure (soft sensor or ratio method)."
function annualization_statement(res::NamedTuple)
    if haskey(res, :lo) && haskey(res, :hi)
        return @sprintf("%s: E = %.1f tCO₂e [%.1f – %.1f] at 95 %% coverage (u_r = %.1f %%)",
                        res.method, res.E, res.lo, res.hi, 100 * res.u_rel)
    end
    @sprintf("%s: E = %.1f tCO₂e (activity ratio %.2f×, no interval — the method assumes a constant factor)",
             res.method, res.E, haskey(res, :ratio) ? res.ratio : NaN)
end
