### A Pluto.jl notebook ###
# v1.0.3
#
# 07 — Neural soft sensor: turning one analyzer week into a year of estimates.

using Markdown
using InteractiveUtils

# ╔═╡ 00000707-0000-0000-0000-000000000001
begin
	import Pkg
	Pkg.activate(normpath(joinpath(@__DIR__, "..")))
end

# ╔═╡ 00000707-0000-0000-0000-000000000002
begin
	using CarbonAccounting, CSV, DataFrames, Dates, Statistics, Printf
	using Plots
	default(fmt=:svg, legend=:topright, size=(760, 420), dpi=120)
	datadir(parts...) = CarbonAccounting.datadir(parts...)
	const TAGS = ["load_pct", "stack_temp_C", "o2_pct", "flare_flow_m3h", "pressure_bar",
	              "feedstock_tph"]
	const ρ_CH₄ = CarbonAccounting.ρ_CH₄          # kg/m³ at normal conditions
end

# ╔═╡ 00000707-0000-0000-0000-000000000003
md"""
# 07 · A neural soft sensor for the annual analyzer campaign

The operational reality: the expensive measurement — a mobile analyzer, a lab
campaign, a certified stack test — happens **once a year for one week**,
`N_camp = 7 d · 24 h = 168` samples, while the plant produces continuous data every
hour. The soft sensor learns the mapping

    ŷ = f_θ⃗(x⃗)      x⃗ : continuous DCS/SCADA tags (load, temperature, O₂, flow, …)
                      y : what the campaign measures (here: CH₄ mole fraction %)

and then predicts `y` for all 8 760 hours of the year — with a quantified
uncertainty instead of a smoothed annual average.
"""

# ╔═╡ 00000707-0000-0000-0000-000000000004
md"""
### The network, written as the equations

    z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾          a⁽ˡ⁾ = g(z⁽ˡ⁾)         ŷ = W⁽ᴸ⁾a⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾
    δ⁽ᴸ⁾ = ∂L/∂ŷ    δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾ ⊙ g′(z⁽ˡ⁾)    ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ + λ_L2 W⁽ˡ⁾

Trained with mini-batch **Adam** (`β₁ = 0.9`, `β₂ = 0.999`) and early stopping on a
validation split. Two output heads make it *heteroscedastic*: it predicts `ŷ` and
`σ̂(x⃗)`, its own noise level, through the Gaussian NLL

    L = ½[(μ − y)²e^{−s} + s],     s = log σ̂²
"""

# ╔═╡ 00000707-0000-0000-0000-000000000005
begin
	campaign = CSV.read(datadir("raw", "analyzer_campaign_2024.csv"), DataFrame)
	dcs = CSV.read(datadir("raw", "dcs_hourly_2024.csv"), DataFrame)
	X = Matrix(campaign[:, TAGS])                     # campaign features
	y = Vector(campaign.ch4_fraction_pct)             # what the analyzer measured
	X_year = Matrix(dcs[:, TAGS])                     # continuous data for the year
	(campaign=size(X), year=size(X_year),
	 tags=DataFrame(tag=TAGS,
	                campaign_min=vec(minimum(X; dims=1)), campaign_max=vec(maximum(X; dims=1)),
	                year_min=vec(minimum(X_year; dims=1)), year_max=vec(maximum(X_year; dims=1))))
end

# ╔═╡ 00000707-0000-0000-0000-000000000006
md"### Train, and look at the convergence — not only at the final number"

# ╔═╡ 00000707-0000-0000-0000-000000000007
begin
	net = train_soft_sensor(X, y; hidden=[16, 8], heteroscedastic=true,
	                        epochs=1200, η=0.01, λ_L2=1e-4, seed=3)
	net
end

# ╔═╡ 00000707-0000-0000-0000-000000000008
begin
	h = training_history(net)
	plt_train = plot(h.epoch, [h.train h.val], label=["train" "validation"],
	           xlabel="epoch", ylabel="loss (NLL, may be negative)", title="Convergence (early stopping)")
	vline!(plt_train, [net.trained_epochs], label="best epoch", ls=:dash, color=:grey)
	save_figure(plt_train, "07_training_history"; saver=savefig)
	plt_train
end

# ╔═╡ 00000707-0000-0000-0000-000000000009
md"""
### Scores that matter: error *and* honesty

RMSE and R² say how close the predictions are; `coverage95` says whether the stated
uncertainty can be trusted. A model with 95 % coverage has error bars that mean
something — the property a compliance team actually needs.
"""

# ╔═╡ 00000707-0000-0000-0000-000000000010
begin
	ŷ = predict(net, X)
	iv = predict_interval(net, X)
	metrics_table(softsensor_metrics(y, ŷ; intervals=iv))
end

# ╔═╡ 00000707-0000-0000-0000-000000000011
cross_validate_softsensor(X, y; k=5, epochs=600, seed=11)

# ╔═╡ 00000707-0000-0000-0000-000000000012
md"""
### An ensemble, so that uncertainty has two components

    σ_total² = σ_epistemic² + σ̂_aleatoric²

The spread between ensemble members is the epistemic part — how much the answer
depends on which week happened to be measured — while the heteroscedastic head gives
the aleatoric part per hour.
"""

# ╔═╡ 00000707-0000-0000-0000-000000000013
begin
	nets = train_ensemble(X, y; n_models=5, epochs=800, heteroscedastic=true, η=0.01)
	ivc = ensemble_interval(nets, X)
	lower = [[i.ŷ - i.lo for i in ivc]...]
	upper = [[i.hi - i.ŷ for i in ivc]...]
	plt_fit = plot(1:length(y), y, label="analyzer (campaign week)", color=:black, lw=2)
	plot!(plt_fit, 1:length(y), [i.ŷ for i in ivc], ribbon=(lower, upper), fillalpha=0.3,
	      label="soft sensor ± 95 %", color=:steelblue)
	save_figure(plt_fit, "07_softsensor_fit"; saver=savefig)
	plt_fit
end

# ╔═╡ 00000707-0000-0000-0000-000000000014
md"""
### Does one week represent the year?

Two tests on the tag matrix: the **envelope** per feature (fraction of annual hours
inside the campaign range) and the **Mahalanobis** distance of each annual hour from
the campaign cloud, compared with the χ² limit — which catches combinations of tags
that are individually plausible but jointly unseen.
"""

# ╔═╡ 00000707-0000-0000-0000-000000000015
begin
	design = campaign_design(X, X_year; feature_names=TAGS,
	                         activity=Vector(dcs.flare_flow_m3h))
	(coverage=round(design.coverage; digits=3),
	 emission_weighted_risk=round(design.risk_share; digits=3),
	 χ²_limit=round(design.chi2_limit; digits=1),
	 recommendation=design.recommendation,
	 envelope=design.features)
end

# ╔═╡ 00000707-0000-0000-0000-000000000016
begin
	needed = min_campaign_samples(0.35; δ=0.05)
	println("to halve the campaign-mean interval: N ≥ ", needed,
	        " samples (a one-week campaign provides 168)")
end

# ╔═╡ 00000707-0000-0000-0000-000000000017
md"### Annualise, and compare with the method it replaces"

# ╔═╡ 00000707-0000-0000-0000-000000000018
begin
	scale = ρ_CH₄ / 100                    # volume-% of CH₄ → kg CH₄ per m³
	annual = annualize_with_softsensor(nets, X_year, Vector(dcs.flare_flow_m3h);
	                                   GWP₁₀₀=GWP_AR6.CH₄_fossil, hours=1.0,
	                                   target_scale=scale)
	week = 1:168
	ratio = annualize_ratio(Vector(dcs.flare_flow_m3h), Vector(campaign.flare_flow_m3h[week]),
	                        scale * sum(Vector(campaign.flare_flow_m3h[week]) .* y[week]) *
	                        GWP_AR6.CH₄_fossil / 1000)
	(soft_sensor=annualization_statement(annual), incumbent=annualization_statement(ratio),
	 difference_pct=round(100 * (annual.E - ratio.E) / ratio.E; digits=1))
end

# ╔═╡ 00000707-0000-0000-0000-000000000019
begin
	per_hour = annual.per_hour
	plt_emit = plot(per_hour.t, per_hour.E_t, label="soft-sensor emission", color=:darkorange,
	           xlabel="hour of 2024", ylabel="tCO₂e per hour", title="Annualised emission")
	plot!(plt_emit, 1:168, per_hour.E_t[1:168], label="campaign week", color=:steelblue, lw=2)
	save_figure(plt_emit, "07_annualised_emission"; saver=savefig)
	plt_emit
end

# ╔═╡ 00000707-0000-0000-0000-000000000020
begin
	println("campaign samples     : ", length(y), " hours")
	println("annual estimate      : ", round(annual.E; digits=1), " tCO₂e [",
	        round(annual.lo; digits=1), " – ", round(annual.hi; digits=1), "]")
	println("relative uncertainty : ", round(100 * annual.u_rel; digits=1), " %")
	println("campaign coverage    : ", round(100 * design.coverage; digits=1), " % of annual hours")
	println("Notebook 07 — the soft sensor reproduces the measured variable, states an honest interval, and shows where the campaign does not represent the year.")
end

