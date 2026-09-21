### A Pluto.jl notebook ###
# v1.0.3
#
# 04 — Scope 1: direct emissions (stationary, mobile, fugitive, flaring, process).

using Markdown
using InteractiveUtils

# ╔═╡ 00000404-0000-0000-0000-000000000001
begin
	import Pkg
	Pkg.activate(normpath(joinpath(@__DIR__, "..")))
end

# ╔═╡ 00000404-0000-0000-0000-000000000002
begin
	using CarbonAccounting, CSV, DataFrames, Dates, Statistics, Printf
	using Plots
	default(fmt=:svg, legend=:topleft, size=(760, 420), dpi=120)
	datadir(parts...) = CarbonAccounting.datadir(parts...)
end

# ╔═╡ 00000404-0000-0000-0000-000000000003
md"""
# 04 · Scope 1 — direct emissions

Five formulas, each with its own physics, all returning `EmissionRow`s:

    E_stat  = Σᵢ Q̇ᵢ/η_b · (EF_CO₂,ᵢ·OFᵢ + GWP_CH₄·EF_CH₄,ᵢ + GWP_N₂O·EF_N₂O,ᵢ)
    E_flare = V̇·EF·DRE·η_comb + GWP_CH₄·V̇·ρ_CH₄·x_CH₄·(1 − DRE·η_comb)
    E_fug   = (C_start + C_purchased − C_end − C_recovered − C_disposed) · GWP₁₀₀
    E_LDAR  = N · leak_rate · GWP_CH₄
    E_proc  = P · EF          (stoichiometry per tonne of product)

The engines never return a bare number: each row carries its factor id, version,
share ©ᵢ, data-quality class and uncertainty.
"""

# ╔═╡ 00000404-0000-0000-0000-000000000004
md"### Stationary combustion, gas by gas — oxidation factor and boiler efficiency"

# ╔═╡ 00000404-0000-0000-0000-000000000005
begin
	Q̇ᵢ = 1.0e5                     # GJ of fuel input
	rows = [(fuel, combustion_emissions(Q̇ᵢ; EF_CO₂=56.1, EF_CH₄=1e-3, EF_N₂O=1e-4,
	                                    OF=1.0, η_b=0.9, fossil=(fuel != "biomass")))
	        for fuel in ["natural_gas", "fuel_oil", "coal"]]
	DataFrame(fuel=[r[1] for r in rows], E=[r[2].E for r in rows],
	          E_CO₂=[r[2].E_CO₂ for r in rows], E_CH₄=[r[2].E_CH₄ for r in rows],
	          E_N₂O=[r[2].E_N₂O for r in rows])
end

# ╔═╡ 00000404-0000-0000-0000-000000000006
md"""
### Flaring is not one multiplication

Only the fraction `φ = DRE·η_comb` of the gas is destroyed; the rest leaves as
**methane**, which at GWP₁₀₀ = 29.8 dominates the result.
"""

# ╔═╡ 00000404-0000-0000-0000-000000000007
begin
	V̇ = 1.0e6                                     # m³(n) of flare gas in the year
	r = flare_emissions(V̇; EF_flare=1.85, DRE=0.98, η_comb=0.995, x_CH₄=0.8)
	(E=r.E, E_combusted=r.E_combusted, E_slip=r.E_slip, φ=r.φ,
	 m_CH₄_slip_kg=r.m_CH₄_slip, note="slip is $(round(100*r.E_slip/r.E; digits=1)) % of the total")
end

# ╔═╡ 00000404-0000-0000-0000-000000000008
md"### Fugitive: refrigerant mass balance and the LDAR component count"

# ╔═╡ 00000404-0000-0000-0000-000000000009
begin
	ref = fugitive_refrigerant(; C_start=1_200, C_purchased=140, C_end=1_305,
	                           C_recovered=25, C_disposed=0, GWP₁₀₀=2088)   # R410A
	ldar = fugitive_ldar(1_850; leak_rate=0.5)
	(refrigerant=ref, ldar=ldar,
	 note="the mass balance finds $(ref.m_lost) kg lost; the LDAR estimate is an independent view")
end

# ╔═╡ 00000404-0000-0000-0000-000000000010
md"### Process emissions: stoichiometry per tonne of product"

# ╔═╡ 00000404-0000-0000-0000-000000000011
begin
	cases = [("clinker", 42_000.0, 0.525, "CO2", "tCO₂/t"),
	         ("lime", 12_000.0, 0.785, "CO2", "tCO₂/t"),
	         ("nitric_acid", 3_100.0, 5.7, "N2O", "kgN₂O/t"),
	         ("ammonia", 8_000.0, 1.90, "CO2e", "tCO₂e/t")]
	DataFrame(product=[c[1] for c in cases], production_t=[c[2] for c in cases],
	          EF=[c[3] for c in cases], unit=[c[5] for c in cases],
	          E_tCO₂e=[round(process_emissions(c[2]; EF=c[3], gas=c[4], unit=c[5]).E; digits=1) for c in cases])
end

# ╔═╡ 00000404-0000-0000-0000-000000000012
md"""
### The portfolio

`compute_scope1` routes every activity record to the right formula of the table
above (see the docstring of `scope1_row`), applies the consolidation share ©ᵢ and
keeps `E_stat`, `E_mob`, `E_fug`, `E_flare`, `E_proc` apart.
"""

# ╔═╡ 00000404-0000-0000-0000-000000000013
begin
	records = ingest_files([datadir("raw", "activity_erp_2024.csv")];
	                       entity_default="AcmeIndustrial_SA").records
	scope1 = compute_scope1(records; consolidation=:operational, date=Date(2024, 12, 31))
	scope1
end

# ╔═╡ 00000404-0000-0000-0000-000000000014
scope1_detail(scope1) |> df -> df[1:min(8, nrow(df)), :]

# ╔═╡ 00000404-0000-0000-0000-000000000015
begin
	parts = sort(collect(scope1.by_source))
	plt = bar([p[1] for p in parts], [p[2] for p in parts], legend=false,
	          title="Scope 1 by source category", ylabel="tCO₂e", color=:darkorange)
	save_figure(plt, "04_scope1_breakdown"; saver=savefig)
	plt
end

# ╔═╡ 00000404-0000-0000-0000-000000000016
begin
	# data quality and uncertainty of each contributing sub-total
	df = emissions_dataframe(scope1.rows)
	sort(combine(groupby(df, :source_type),
	             :E => sum => :E, :DQ => (x -> round(mean(x); digits=2)) => :mean_DQ,
	             :u_rel => (x -> round(mean(x); digits=3)) => :mean_u_r), :E, rev=true)
end

# ╔═╡ 00000404-0000-0000-0000-000000000017
begin
	println("E_Scope1         : ", round(scope1.E_Scope1; digits=1), " tCO₂e")
	println("biogenic memo    : ", round(scope1.memo_biogenic; digits=1), " tCO₂e (excluded from the total)")
	println("rows             : ", length(scope1.rows))
	println("Notebook 04 — combustion, flaring, fugitive and process emissions are computed by their own physics, not by one generic factor.")
end

