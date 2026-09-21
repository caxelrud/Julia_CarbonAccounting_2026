### A Pluto.jl notebook ###
# v1.0.3
#
# 03 — Emission-factor libraries: provenance, versions, precedence, dimensional fit.

using Markdown
using InteractiveUtils

# ╔═╡ 00000303-0000-0000-0000-000000000001
begin
	import Pkg
	Pkg.activate(normpath(joinpath(@__DIR__, "..")))
end

# ╔═╡ 00000303-0000-0000-0000-000000000002
begin
	using CarbonAccounting, CSV, DataFrames, Dates, Statistics, Printf
	datadir(parts...) = CarbonAccounting.datadir(parts...)
	figdir(parts...) = CarbonAccounting.figdir(parts...)
end

# ╔═╡ 00000303-0000-0000-0000-000000000003
md"""
# 03 · Emission-factor libraries

A factor is not a number, it is a **claim**: it holds for a given activity, region,
gas and time window, somebody published it, and it carries an uncertainty. Every
record is therefore stored as an immutable `EFRecord` with

    EFᵢ = (value, unit, gas, region, GWP set, validity window, DQ, σ/μ, source, version)

and resolved for an activity by (region → GLOBAL, newest version, item match).
Changing a factor creates a **new version**; it never overwrites history.
"""

# ╔═╡ 00000303-0000-0000-0000-000000000004
begin
	lib = default_library()
	(count=length(lib), name=lib.name, version=lib.version,
	 categories=length(unique(f.category for f in lib.records)),
	 sources=length(unique(f.source for f in lib.records)))
end

# ╔═╡ 00000303-0000-0000-0000-000000000005
begin
	df = factors_dataframe(lib)
	first(select(df, :category, :item, :gas, :value, :unit, :region, :version, :DQ, :uncertainty), 8)
end

# ╔═╡ 00000303-0000-0000-0000-000000000006
md"""
### Precedence: the same factor id, several claims

`lookup_factor` returns the candidates in precedence order — the requested region
first, then `GLOBAL`, then any other region, newest version inside each group. The
order is explicit, so a report can state *which* factor set it used.
"""

# ╔═╡ 00000303-0000-0000-0000-000000000007
begin
	# a factor that exists in several regions and versions
	extras = [EFRecord("grid_DEU_2025.1", "Utility disclosure 2025", "2025.1", "electricity",
	                   "grid_average", "CO2e", 0.352, "kgCO₂e/kWh", "DEU", "AR5",
	                   Date(2025, 1, 1), Date(2026, 12, 31), 2, 0.04, "utility residual disclosure", ""),
	          EFRecord("grid_BRA_2025.1", "Utility disclosure 2025", "2025.1", "electricity",
	                   "grid_average", "CO2e", 0.098, "kgCO₂e/kWh", "BRA", "AR5",
	                   Date(2025, 1, 1), Date(2026, 12, 31), 2, 0.06, "utility residual disclosure", "")]
	lib2 = FactorLibrary(lib.name, "2026.2", vcat(lib.records, extras))
	(rows=[(f.region, f.version, f.value, f.source) for f in
	       lookup_factor(lib2, "electricity", "grid_average"; region="DEU", date=Date(2025, 6, 1))],
	 note="DEU 2025.1 wins; the 2026.1 GLOBAL default follows")
end

# ╔═╡ 00000303-0000-0000-0000-000000000008
md"""
### Dimensional fit is part of factor selection

`compatibility(f, activity_unit)` refuses to pair a factor with an activity of the
wrong dimension — so a `kgCO₂e/kWh` factor never meets a `GJ` activity.
"""

# ╔═╡ 00000303-0000-0000-0000-000000000009
begin
	f = lookup_factor(lib, "natural_gas", "natural_gas"; region="GLOBAL")[1]
	describe_factor(f), compatibility(f, "GJ"), compatibility(f, "MWh"), compatibility(f, "t")
end

# ╔═╡ 00000303-0000-0000-0000-000000000010
md"""
### Resolution: free-text activity → factor

`resolve_factor` walks the rule: exact item, then the rule's default item, then an
EEIO sector hint from the procurement text, and finally the mobile-combustion set.
The same entry point serves Scope 1, 2 and 3.
"""

# ╔═╡ 00000303-0000-0000-0000-000000000011
begin
	rules = ["natural_gas", "refrigerant", "ldar", "flaring", "process_cement",
	         "electricity", "waste", "upstream_transport"]
	items = Dict("refrigerant" => "R410A", "ldar" => nothing, "flaring" => nothing,
	             "process_cement" => "clinker", "electricity" => nothing,
	             "waste" => "recycling", "upstream_transport" => "sea_container",
	             "natural_gas" => nothing)
	DataFrame(source_type=rules,
	          chosen_factor=[resolve_factor(lib, SOURCE_RULES[r];
	                           item=get(items, r, nothing), region="GLOBAL",
	                           date=Date(2024, 12, 31)) |> f -> f === nothing ? "—" : f.ef_id
	                      for r in rules])
end

# ╔═╡ 00000303-0000-0000-0000-000000000012
md"""
### Spend-based factors: from purchase-order text to an EEIO sector

`purchase → sector_hint → EF_EEIO`, with the spend deflated to the base year of the
factor set.
"""

# ╔═╡ 00000303-0000-0000-0000-000000000013
begin
	texts = ["stainless steel plate", "facility management", "corrugated packaging",
	         "cold-chain logistics", "agricultural fertiliser"]
	idx = PriceIndex(2024, Dict(2020 => 100.0, 2021 => 104.7, 2022 => 113.2, 2023 => 117.9,
	                            2024 => 121.9))
	DataFrame(text=texts, sector=sector_hint.(texts),
	          spend_2023_deflated=[round(deflate(1_000_000, 2023, idx); digits=0) for _ in texts])
end

# ╔═╡ 00000303-0000-0000-0000-000000000014
md"""
### Every factor carries its own uncertainty and quality

That is what lets the uncertainty engine (notebook 10) propagate a *defensible*
budget instead of a blanket percentage.
"""

# ╔═╡ 00000303-0000-0000-0000-000000000015
combine(groupby(factors_dataframe(lib), :category),
        :uncertainty => (x -> round(100 * mean(x); digits=1)) => :mean_u_r_pct, nrow => :records)

# ╔═╡ 00000303-0000-0000-0000-000000000016
begin
	out = joinpath(CarbonAccounting.builddir("notebook03"), "factors_snapshot.csv")
	export_factors(lib, out)
	reloaded = load_factors(out; name="snapshot")
	println("exported ", length(lib), " factors to ", out)
	println("reloaded ", length(reloaded), " factors, version ", reloaded.version)
	println("Notebook 03 — the factor set is versioned, provenance-documented, dimension-checked and exportable.")
end

