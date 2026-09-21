### A Pluto.jl notebook ###
# v1.0.3
#
# 05 — Scope 2: purchased energy, reported twice (location and market based).

using Markdown
using InteractiveUtils

# ╔═╡ 00000505-0000-0000-0000-000000000001
begin
	import Pkg
	Pkg.activate(normpath(joinpath(@__DIR__, "..")))
end

# ╔═╡ 00000505-0000-0000-0000-000000000002
begin
	using CarbonAccounting, CSV, DataFrames, Dates, Statistics, Printf
	using Plots
	default(fmt=:svg, legend=:topleft, size=(760, 420), dpi=120)
	datadir(parts...) = CarbonAccounting.datadir(parts...)
end

# ╔═╡ 00000505-0000-0000-0000-000000000003
md"""
# 05 · Scope 2 — purchased electricity, steam, heat and cooling

The GHG Protocol requires the **dual report**:

    E_loc = Σᵢ ( Cᵢ · EF_grid,y )                                  location based
    E_mkt = Σᵢ ( Cᵢ · EF_contract ) + C_unclaimed · EF_residual    market based

`compute_scope2` emits *two rows per purchase* — `location-based` and
`market-based` — with the same activity, share ©ᵢ and uncertainty, so both figures
decompose identically. The market factor comes from the contractual instrument
declared on the record (PPA, green tariff), otherwise the residual mix applies;
λ_cov states how much of the consumption instruments cover at all.
"""

# ╔═╡ 00000505-0000-0000-0000-000000000004
md"### Where the market-based factor comes from — the contractual hierarchy"

# ╔═╡ 00000505-0000-0000-0000-000000000005
begin
	lib = default_library()
	ppa = resolve_factor(lib, SOURCE_RULES["ppa_electricity"]; item="ppa_renewable",
	                     region="GLOBAL", date=Date(2024, 12, 31)).value
	tariff = resolve_factor(lib, SOURCE_RULES["ppa_electricity"]; item="green_tariff",
	                        region="GLOBAL", date=Date(2024, 12, 31)).value
	resid = resolve_factor(lib, SOURCE_RULES["electricity"]; region="GLOBAL",
	                       date=Date(2024, 12, 31), market=:market).value
	DataFrame(instrument=["PPA, renewable", "Green tariff with certificates",
	                      "No instrument → residual mix"],
	          factor=[ppa, tariff, resid], unit="kgCO₂e/kWh")
end

# ╔═╡ 00000505-0000-0000-0000-000000000006
md"### Location-based grid intensity versus residual mix, by country"

# ╔═╡ 00000505-0000-0000-0000-000000000007
begin
	countries = ["BRA", "DEU", "ZAF", "CHN", "GLOBAL"]
	loc = [resolve_factor(lib, SOURCE_RULES["electricity"]; region=c,
	                      date=Date(2024, 12, 31)).value for c in countries]
	res = [resolve_factor(lib, SOURCE_RULES["electricity"]; region=c,
	                      date=Date(2024, 12, 31), market=:market).value for c in countries]
	plt_grid = bar(countries, [loc res], bar_position=:dodge,
	                 label=["location based" "residual mix (market)"],
	                 title="Grid intensity vs residual mix", ylabel="kgCO₂e/kWh")
	save_figure(plt_grid, "05_grid_vs_residual"; saver=savefig)
	DataFrame(country=countries, EF_grid=loc, EF_residual=res,
	          premium_pct=[round(100 * (r / l - 1); digits=1) for (l, r) in zip(loc, res)])
end

# ╔═╡ 00000505-0000-0000-0000-000000000008
md"### The purchases of the portfolio — mixed units in one file, instruments per month"

# ╔═╡ 00000505-0000-0000-0000-000000000009
begin
	records = ingest_files([datadir("raw", "electricity_utility_2024.csv")];
	                       entity_default="AcmeIndustrial_SA").records
	df = records_dataframe(records)
	(purchases=nrow(df), by_unit=combine(groupby(df, [:unit, :unit_canon]), nrow => :n),
	 instruments=unique([r.meta["instrument"] for r in records]))
end

# ╔═╡ 00000505-0000-0000-0000-000000000010
md"### The dual report"

# ╔═╡ 00000505-0000-0000-0000-000000000011
compute_scope2(records; date=Date(2024, 12, 31), consolidation=:operational)

# ╔═╡ 00000505-0000-0000-0000-000000000012
begin
	scope2 = compute_scope2(records; date=Date(2024, 12, 31), consolidation=:operational)
	detail = scope2_detail(scope2)
	first(select(detail, :site, :item, :method, :Q̇, :unit, :ef_id, :ef_value, :ef_unit, :E, :DQ), 8)
end

# ╔═╡ 00000505-0000-0000-0000-000000000013
begin
	per_site = combine(groupby(scope2_detail(scope2), [:site, :method]),
	                   :Q̇ => sum => :activity_GJ, :E => sum => :E)
	sort!(per_site, [:site, :method])
	per_site
end

# ╔═╡ 00000505-0000-0000-0000-000000000014
md"### Dual reporting is not a formality: the two figures answer different questions"

# ╔═╡ 00000505-0000-0000-0000-000000000015
begin
	plt_dual = bar(["location based", "market based"],
	               reshape([scope2.E_loc, scope2.E_mkt], 2, 1),
	               bar_position=:dodge,
	                 label="", legend=false, ylabel="tCO₂e",
	                 title="Scope 2 of the portfolio (operational control)",
	                 color=[:seagreen :darkseagreen])
	save_figure(plt_dual, "05_scope2_dual"; saver=savefig)
	println("E_Scope2 location : ", round(scope2.E_loc; digits=1), " tCO₂e")
	println("E_Scope2 market   : ", round(scope2.E_mkt; digits=1), " tCO₂e")
	println("purchased electricity : ", round(scope2.C_elec; digits=1), " MWh")
	println("contractual coverage λ_cov : ", round(100 * scope2.λ_cov; digits=1), " %")
	println("Notebook 05 — both Scope 2 figures come from the same activities, so their difference is attributable.")
end

