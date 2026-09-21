# ══════════════════════════════════════════════════════════════════════════════
#  uncertainty.jl — how uncertain is the inventory?
#
#  Two independent routes, as ISO/IEC Guide 98-3 (GUM) and the GHG Protocol
#  recommend:
#
#  1. **GUM / sensitivity propagation.** For E = f(x₁,…,xₙ),
#
#         u_c² = Σᵢ (cᵢ σᵢ)² + 2 ΣΣ_{i<j} ρᵢⱼ cᵢ cⱼ σᵢ σⱼ,     cᵢ = ∂E/∂xᵢ
#
#     For a pure product Eᵢ = Q̇ᵢ·EFᵢ the sensitivity coefficients are the factors
#     themselves, so the relative uncertainties add in quadrature:
#
#         (u_E/E)² = (σ_activity/Q̇)² + (σ_factor/EF)²
#
#  2. **Monte Carlo.** Draw each input from its distribution (normal, lognormal,
#     triangular, uniform), recompute the whole inventory, and take the empirical
#     2.5 % / 97.5 % quantiles. This is the only honest route when the model is
#     non-linear (flaring, mass balances with subtraction, screening thresholds).
# ══════════════════════════════════════════════════════════════════════════════

"""
    UncertaintySpec(dist, u_rel, params)

Distribution of one input: `dist ∈ {:normal, :lognormal, :triangular, :uniform}`,
`u_rel` the relative standard uncertainty σ/μ (1σ) and `params` extra entries
(e.g. `lo`, `mode`, `hi` for a triangular judgement).
"""
Base.@kwdef struct UncertaintySpec
    dist::Symbol = :normal
    u_rel::Float64 = 0.05
    params::Dict{Symbol,Float64} = Dict{Symbol,Float64}()
    note::String = ""
end

gaussian(u_rel::Real; note::String="") = UncertaintySpec(dist=:normal, u_rel=Float64(u_rel), note=note)
lognormal(u_rel::Real; note::String="") = UncertaintySpec(dist=:lognormal, u_rel=Float64(u_rel), note=note)
triangular(lo::Real, mode::Real, hi::Real; note::String="") =
    UncertaintySpec(dist=:triangular, u_rel=(hi - lo) / (2 * sqrt(6) * max(abs(mode), eps())),
                    params=Dict(:lo => Float64(lo), :mode => Float64(mode), :hi => Float64(hi)), note=note)
uniform_unc(lo::Real, hi::Real; note::String="") =
    UncertaintySpec(dist=:uniform, u_rel=(hi - lo) / (2 * sqrt(3) * max(abs((hi + lo) / 2), eps())),
                    params=Dict(:lo => Float64(lo), :hi => Float64(hi)), note=note)

"""
    sample(spec, μ, rng) -> Float64

Draw one realisation of an input with mean μ. Lognormal and normal draws are
specified by their *relative* standard uncertainty, which is how factor and
activity uncertainties are actually reported.
"""
function sample(spec::UncertaintySpec, μ::Real, rng::AbstractRNG=Random.default_rng())
    σ = spec.u_rel * abs(μ)
    if spec.dist == :normal
        return μ + σ * randn(rng)
    elseif spec.dist == :lognormal
        σ_log = sqrt(log1p(spec.u_rel^2))
        μ_log = log(max(abs(μ), eps())) - σ_log^2 / 2
        return sign(μ) * exp(μ_log + σ_log * randn(rng))
    elseif spec.dist == :triangular
        lo = get(spec.params, :lo, μ * (1 - 3 * spec.u_rel))
        mode = get(spec.params, :mode, μ)
        hi = get(spec.params, :hi, μ * (1 + 3 * spec.u_rel))
        s = sqrt(rand(rng))
        f = (mode - lo) / max(hi - lo, eps())
        return rand(rng) < f ? lo + s * (mode - lo) * sqrt(f) : hi - s * (hi - mode) * sqrt(1 - f)
    elseif spec.dist == :uniform
        lo = get(spec.params, :lo, μ * (1 - 2 * spec.u_rel))
        hi = get(spec.params, :hi, μ * (1 + 2 * spec.u_rel))
        return lo + (hi - lo) * rand(rng)
    end
    throw(ArgumentError("unknown uncertainty distribution «$(spec.dist)»"))
end

# ── GUM: sensitivity propagation ─────────────────────────────────────────────
"Uncertainty specification of one row: explicit override, else its own σ/μ."
u_of(r::EmissionRow, specs::AbstractDict) = haskey(specs, r.rec_id) ? specs[r.rec_id].u_rel : r.u_rel

"""
    UncertaintySummary

Result of an uncertainty calculation: central value, combined standard
uncertainty `u_c`, relative form, expanded `U₉₅ = 2·u_c`, empirical quantiles and
the raw Monte-Carlo sample (kept for histograms).
"""
struct UncertaintySummary
    E::Float64
    u_c::Float64
    u_rel::Float64
    U95::Float64
    lo95::Float64
    hi95::Float64
    n_MC::Int
    method::String
    quantiles::Vector{Float64}
    samples::Vector{Float64}
end

function Base.show(io::IO, s::UncertaintySummary)
    @printf(io, "%s: E = %.3f tCO₂e, u_c = %.3f (%.2f%%), U₉₅ = %.3f, 95%% CI = [%.3f, %.3f] (n=%d)",
            s.method, s.E, s.u_c, 100 * s.u_rel, s.U95, s.lo95, s.hi95, s.n_MC)
end

"""
    gum_uncertainty(rows; specs, rho_factor, group_by) -> UncertaintySummary

ISO/IEC Guide 98-3 propagation, written in the *structure of the model* rather
than in the totals. Each row carries two uncertainty components — the activity
(`u_act`, independent between rows) and the factor (`u_fac`, shared inside a factor
group, i.e. the same published claim) — so that

    E = Σ_g F_g · A_g ,  A_g = Σ_{i∈g} Eᵢ

gives

    u_c² = Σ_g [ Σ_{i∈g} (Eᵢ u_act,ᵢ)² + (A_g u_fac,g)² ] + 2ρ Σ_{g<h} A_g A_h u_fac,g u_fac,h

The decomposition is O(n + G²), needs no division by a group total (which can be
zero when a credit offsets an emission) and makes the correlation assumption
explicit: rows sharing a factor move together with coefficient ρ.
"""
function gum_uncertainty(rows::AbstractVector{EmissionRow};
                         specs::AbstractDict=Dict{String,UncertaintySpec}(),
                         rho_factor::Real=0.80,
                         group_by::Function=r -> (r.library, r.version, r.ef_id))
    groups = Dict{Any,Vector{Int}}()
    for (i, r) in enumerate(rows)
        push!(get!(groups, group_by(r), Int[]), i)
    end
    u_act_of(r) = haskey(specs, r.rec_id) ? specs[r.rec_id].u_rel :
                  (r.u_act > 0 ? r.u_act : r.u_rel / sqrt(2))
    u_fac_of(r) = r.u_fac > 0 ? r.u_fac : r.u_rel / sqrt(2)

    u_c2 = 0.0
    A, U = Float64[], Float64[]
    for (_, idxs) in groups
        A_g = Σᵢ(i -> rows[i].E, idxs)
        u_f = mean(u_fac_of(rows[i]) for i in idxs)
        push!(A, A_g)
        push!(U, u_f)
        for i in idxs
            u_c2 += (rows[i].E * u_act_of(rows[i]))^2
        end
        u_c2 += (A_g * u_f)^2
    end
    if rho_factor != 0
        for i in 1:length(A), j in (i + 1):length(A)
            u_c2 += 2 * rho_factor * A[i] * A[j] * U[i] * U[j]
        end
    end
    E_tot = Σᵢ(r -> r.E, rows)
    u_c = sqrt(max(u_c2, 0.0))
    UncertaintySummary(E_tot, u_c, E_tot > 0 ? u_c / E_tot : 0.0, 2 * u_c,
                       max(E_tot - 2 * u_c, 0.0), E_tot + 2 * u_c, 0, "GUM quadrature",
                       Float64[], Float64[])
end

# ── Monte Carlo ──────────────────────────────────────────────────────────────
"""
    monte_carlo_uncertainty(rows; n_MC, seed, rho_factor, activity_spec, factor_spec, specs) -> UncertaintySummary

Draw a scenario `n_MC` times and recompute the inventory total each time:

    Q̇ᵢ(s) ~ activity distribution,   EFᵢ(s) ~ factor distribution  (shared inside a factor group)
    E(s)   = Σᵢ Q̇ᵢ(s) · EFᵢ(s)

Because factor draws are shared by group, the simulation reproduces the
correlated part of the uncertainty that the GUM formula approximates. The
empirical 2.5 % / 97.5 % quantiles give the 95 % interval — which, for the
lognormal-shaped factor distributions that dominate real inventories, is *not*
symmetric around the central value.

`rows` may also be the rows of an inventory from any scope; the engine is
agnostic about where they came from.
"""
function monte_carlo_uncertainty(rows::AbstractVector{EmissionRow};
                                 n_MC::Integer=5_000, seed::Integer=42,
                                 rho_factor::Real=0.80,
                                 specs::AbstractDict=Dict{String,UncertaintySpec}(),
                                 group_by::Function=r -> (r.library, r.version, r.ef_id))
    rng = MersenneTwister(seed)
    groups = Dict{Any,Vector{Int}}()
    for (i, r) in enumerate(rows)
        push!(get!(groups, group_by(r), Int[]), i)
    end
    u_act_of(r) = haskey(specs, r.rec_id) ? specs[r.rec_id].u_rel :
                  (r.u_act > 0 ? r.u_act : r.u_rel / sqrt(2))
    u_fac_of(r) = r.u_fac > 0 ? r.u_fac : r.u_rel / sqrt(2)
    samples = Vector{Float64}(undef, n_MC)
    # One-factor correlation model: within a factor group the factor perturbation is
    # shared exactly, and between groups it is correlated with coefficient rho —
    #      F_g = 1 + u_fac,g · ( √ρ · ξ_common + √(1−ρ) · ξ_g )
    for s in 1:n_MC
        ξ_common = randn(rng)
        E = 0.0
        for (g, idxs) in groups
            u_f = mean(u_fac_of(rows[i]) for i in idxs)
            ξ_g = randn(rng)
            F = 1.0 + u_f * (sqrt(rho_factor) * ξ_common + sqrt(1 - rho_factor) * ξ_g)
            sub = 0.0
            for i in idxs
                u_a = u_act_of(rows[i])
                sub += rows[i].E * max(1.0 + u_a * randn(rng), 0.0)
            end
            E += max(F, 0.0) * sub
        end
        samples[s] = E
    end
    E_tot = Σᵢ(r -> r.E, rows)
    u_c = sqrt(var(samples))
    qs = quantile(samples, [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975])
    UncertaintySummary(E_tot, u_c, E_tot > 0 ? u_c / E_tot : 0.0, 2 * u_c, qs[1], qs[end],
                       Int(n_MC), "Monte Carlo (n=$(Int(n_MC)))", qs, samples)
end

"""
    monte_carlo_uncertainty(draw::Function; n_MC, seed) -> UncertaintySummary

Generic form for models that are *not* linear in their inputs: `draw(rng)` must
return one complete scenario value (e.g. re-running the flare or soft-sensor
model with perturbed parameters). Use this whenever a single multiplication is
not an honest description of the chain.
"""
function monte_carlo_uncertainty(draw::Function; n_MC::Integer=5_000, seed::Integer=42)
    rng = MersenneTwister(seed)
    samples = [Float64(draw(rng)) for _ in 1:n_MC]
    E_tot = mean(samples)
    u_c = sqrt(var(samples))
    qs = quantile(samples, [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975])
    UncertaintySummary(E_tot, u_c, E_tot > 0 ? u_c / E_tot : 0.0, 2 * u_c, qs[1], qs[end],
                       Int(n_MC), "Monte Carlo (simulation, n=$(Int(n_MC)))", qs, samples)
end

"""
    uncertainty_table(gum, mc) -> DataFrame

Side-by-side report of the two routes — the form an assurance provider asks for:
the GUM figure is analytic and reproducible by hand, the Monte-Carlo figure
captures asymmetry and non-linearity.
"""
function uncertainty_table(gum::UncertaintySummary, mc::Union{Nothing,UncertaintySummary}=nothing)
    rows = [("GUM quadrature (analytic)", gum.E, gum.u_c, gum.u_rel, gum.U95, gum.lo95, gum.hi95, gum.n_MC)]
    if mc !== nothing
        push!(rows, ("Monte Carlo simulation", mc.E, mc.u_c, mc.u_rel, mc.U95, mc.lo95, mc.hi95, mc.n_MC))
        push!(rows, ("Monte Carlo, median", mc.E, mc.u_c, mc.u_rel, mc.U95,
                     mc.quantiles[1], mc.quantiles[end], mc.n_MC))
    end
    DataFrame(method=[r[1] for r in rows], E=[r[2] for r in rows], u_c=[r[3] for r in rows],
              u_rel=[r[4] for r in rows], U95=[r[5] for r in rows], lo95=[r[6] for r in rows],
              hi95=[r[7] for r in rows], n=[r[8] for r in rows])
end

"Disclosure sentence for the report: the 95 % interval of the inventory total."
function uncertainty_statement(s::UncertaintySummary)
    @sprintf("%s: E = %.1f tCO₂e ± %.1f tCO₂e (U₉₅, k = 2), i.e. ±%.1f %%; 95 %% interval [%.1f, %.1f] tCO₂e.",
             s.method, s.E, s.U95, 100 * s.u_rel, s.lo95, s.hi95)
end
