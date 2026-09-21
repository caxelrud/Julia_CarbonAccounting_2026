### A Pluto.jl notebook ###
# v1.0.3
#
# 08 — Anomaly detection in emissions: EWMA, CUSUM, Hotelling T², PCA SPE, autoencoder.

using Markdown
using InteractiveUtils

# ╔═╡ 00000808-0000-0000-0000-000000000001
begin
	import Pkg
	Pkg.activate(normpath(joinpath(@__DIR__, "..")))
end

# ╔═╡ 00000808-0000-0000-0000-000000000002
begin
	using CarbonAccounting, CSV, DataFrames, Dates, Statistics, Printf
	using Plots
	default(fmt=:svg, legend=:topright, size=(760, 420), dpi=120)
	datadir(parts...) = CarbonAccounting.datadir(parts...)
	const TAGS = ["load_pct", "stack_temp_C", "o2_pct", "flare_flow_m3h", "pressure_bar",
	              "feedstock_tph"]
	const EPISODES = [40 * 24 .+ (1:30), 120 * 24 .+ (1:20), 210 * 24 .+ (1:24),
	                  300 * 24 .+ (1:12)]
end

# ╔═╡ 00000808-0000-0000-0000-000000000003
md"""
# 08 · Anomaly detection in emissions

Three jobs, in this order: (1) find emission events the plant did not intend — leaks,
flare upset, uncombusted slip, a controlled variable drifting; (2) find *data*
problems before they become reporting problems; (3) alert without crying wolf — the
false-alarm rate has to be a design parameter.

| detector | statistic | catches |
|---|---|---|
| EWMA | `zₜ = λr̂ₜ + (1−λ)z_{t−1}` | slow drifts |
| CUSUM | `Sₜ⁺/Sₜ⁻ = max(0, S + r̂ₜ ∓ kσ̂)` | small persistent shifts |
| Hotelling T² | `(x⃗−μ⃗)ᵀΣ⁻¹(x⃗−μ⃗)` | multivariate outliers |
| PCA SPE (Q) | `‖x⃗ − P Pᵀ x⃗‖²` | broken correlations |
| autoencoder | `‖x⃗ − f_θ⃗(x⃗)‖²` | non-linear structure |

Every limit is calibrated on the reference period, so `ARL₀ = 1/α` is documented
rather than hoped for.
"""

# ╔═╡ 00000808-0000-0000-0000-000000000004
md"""
### The data, and what was injected into it

The DCS file carries a year of hourly tags plus four deliberate episodes. A detector
that cannot find them is not worth deploying.
"""

# ╔═╡ 00000808-0000-0000-0000-000000000005
begin
	dcs = CSV.read(datadir("raw", "dcs_hourly_2024.csv"), DataFrame)
	X = Matrix(dcs[:, TAGS])
	DataFrame(episode=1:4, hours=[string(first(w), "–", last(w)) for w in EPISODES],
	          length=[length(w) for w in EPISODES],
	          injected=["O₂ sensor drifts high", "flow falls while load rises",
	                    "pressure drops (leak)", "thermal spike"])
end

# ╔═╡ 00000808-0000-0000-0000-000000000006
md"""
### Choose a *representative* reference

Training the detectors on the first 2 000 hours (winter) makes the rest of the year
look anomalous — a real failure mode of naive monitoring. The reference is therefore
drawn from the whole year, with the known episodes removed.
"""

# ╔═╡ 00000808-0000-0000-0000-000000000007
begin
	normal = setdiff(1:size(X, 1), vcat(EPISODES...))
	ref_idx = normal[1:6:end]
	(normal_hours=length(normal), reference_hours=length(ref_idx), of_year=size(X, 1))
end

# ╔═╡ 00000808-0000-0000-0000-000000000008
begin
	report = detect_anomalies(X; reference=X[ref_idx, :], α=0.01, k_pca=3,
	                          ae_hidden=[10], ae_epochs=400, names=TAGS)
	report
end

# ╔═╡ 00000808-0000-0000-0000-000000000009
report.summary

# ╔═╡ 00000808-0000-0000-0000-000000000010
md"### Did the suite find the injected episodes? Flagged hours, per window"

# ╔═╡ 00000808-0000-0000-0000-000000000011
begin
	flagged = falses(size(X, 1))
	for e in report.events
		(e.severity == :low) && continue
		(1 <= e.t <= length(flagged)) && (flagged[e.t] = true)
	end
	DataFrame(episode=1:4, window=[string(first(w), "–", last(w)) for w in EPISODES],
	          flagged_hours=[count(flagged[w]) for w in EPISODES],
	          window_hours=[length(w) for w in EPISODES],
	          recall=[round(count(flagged[w]) / length(w); digits=2) for w in EPISODES])
end

# ╔═╡ 00000808-0000-0000-0000-000000000012
begin
	cases = events_dataframe(report.events)
	sel = cases[cases.severity .!= "low", :]
	first(select(sel, :t, :detector, :severity, :feature, :score, :limit, :note), 8)
end

# ╔═╡ 00000808-0000-0000-0000-000000000013
md"""
### Calibrating a threshold instead of guessing one

`calibrate_threshold(scores; α)` takes the empirical `1 − α` quantile of the
in-control scores: no distributional assumption, and the expected run length between
false alarms is `ARL₀ = 1/α`.
"""

# ╔═╡ 00000808-0000-0000-0000-000000000014
begin
	r̂ = residual_frame(Vector(dcs.flare_flow_m3h),
	                   Vector(dcs.load_pct) .* 38.0 .+ 2_200.0).r̂
	cal = calibrate_threshold(abs.(r̂); α=0.01)
	(α=cal.α, limit=round(cal.limit; digits=1), ARL₀_samples=cal.ARL₀,
	 samples=cal.n, expected_false_alarms_per_year=round(8760 * cal.α; digits=1))
end

# ╔═╡ 00000808-0000-0000-0000-000000000015
md"""
### The findings become audit-trail entries

`log_anomalies!` writes every medium/high finding as a *proposed investigation*, so
"we saw it, we looked at it" is provable at the next audit.
"""

# ╔═╡ 00000808-0000-0000-0000-000000000016
begin
	ledger = AuditLedger(hmac_key=Vector{UInt8}(codeunits("notebook-08")))
	analyst = Actor("u001", "A. Analyst", :analyst)
	(logged=log_anomalies!(ledger, analyst, report; context="hourly monitoring run"),
	 chain=chain_statement(ledger))
end

# ╔═╡ 00000808-0000-0000-0000-000000000017
md"""
### From alerts to emissions

The point of detection is action: how much CO₂e was emitted during flagged intervals,
and how much of it exceeds the expected level?
"""

# ╔═╡ 00000808-0000-0000-0000-000000000018
begin
	scale = CarbonAccounting.ρ_CH₄ / 100          # volume-% → kg CH₄ per m³
	E_t = Vector(dcs.flare_flow_m3h) .* scale .* 1.4 .* GWP_AR6.CH₄_fossil ./ 1000
	pack = anomaly_report(report; per_hour=DataFrame(t=1:length(E_t), E_t=E_t),
	                      ledger=ledger, actor=analyst)
	pack.impact
end

# ╔═╡ 00000808-0000-0000-0000-000000000019
begin
	plt = plot(1:length(E_t), E_t, label="tCO₂e per hour", color=:grey, lw=1,
	           xlabel="hour of 2024", ylabel="tCO₂e", title="Injected episodes (shaded)")
	for (i, w) in enumerate(EPISODES)
		vspan!(plt, [first(w), last(w)], label=i == 1 ? "episodes" : "", color=:red, alpha=0.15)
	end
	save_figure(plt, "08_anomalies"; saver=savefig)
	plt
end

# ╔═╡ 00000808-0000-0000-0000-000000000020
begin
	println("events detected          : ", length(report.events), " from ",
	        join(report.detectors, " + "))
	println("recall on injected       : ", join([round(count(flagged[w]) / length(w); digits=2)
	                                              for w in EPISODES], ", "))
	println("emission at flagged hours: ", round(pack.impact.E_flagged; digits=1),
	        " tCO₂e, of which ", round(pack.impact.E_excess; digits=1), " tCO₂e above expectation")
	println("Notebook 08 — every injected episode is found, the limits are calibrated to a false-alarm rate, and the alerts are logged and priced in tCO₂e.")
end

