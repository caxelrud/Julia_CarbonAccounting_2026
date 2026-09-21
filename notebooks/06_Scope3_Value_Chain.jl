### A Pluto.jl notebook ###
# v1.0.3
#
# 06 — Scope 3: the 15 categories, hybrid methods and the hotspot screening.

using Markdown
using InteractiveUtils

# ╔═╡ 00000606-0000-0000-0000-000000000001
begin
	import Pkg
	Pkg.activate(normpath(joinpath(@__DIR__, "..")))
end

# ╔═╡ 00000606-0000-0000-0000-000000000002
begin
	using CarbonAccounting, CSV, DataFrames, Dates, Statistics, Printf
	using Plots
	default(fmt=:svg, legend=:topleft, size=(760, 420), dpi=120)
	datadir(parts...) = CarbonAccounting.datadir(parts...)
end

# ╔═╡ 00000606-0000-0000-0000-000000000003
md"""
# 06 · Scope 3 — value chain

Scope 3 is where an inventory is made or lost: 15 categories, most of them outside
the company's own meters. The engine applies the **hybrid data hierarchy** of the
GHG Protocol:

| method | activity | factor | used for |
|---|---|---|---|
| supplier specific | supplier-reported | supplier factor | DQ 1 |
| average data | mass, energy, t·km, p·km | industry average | DQ 2–3 |
| spend based | deflated spend | EEIO factor | DQ 4, screening |

Screening is deliberate: the Pareto hotspots carrying ~80 % of the total are exactly
the ones that must be refined with better data — and the engine names them.
"""

# ╔═╡ 00000606-0000-0000-0000-000000000004
md"### The 15 categories and the activity each one expects"

# ╔═╡ 00000606-0000-0000-0000-000000000005
begin
	rows = [(c, SCOPE3_CATEGORY_NAMES[c], r.key, string(r.dim), r.unit_hint)
	        for r in values(SOURCE_RULES) for c in [r.cat]
	        if r.scope == :scope3 && c !== nothing]
	sort!(DataFrame(cat=[r[1] for r in rows], category=[r[2] for r in rows],
	                source_type=[r[3] for r in rows], dimension=[r[4] for r in rows],
	                unit=[r[5] for r in rows]), :cat)
end

# ╔═╡ 00000606-0000-0000-0000-000000000006
md"""
### Spend before deflation is not spend after it

An EEIO factor is expressed per unit of spend *in its own base year*, so a 2023
invoice is deflated to USD2024 before it meets the factor:
`s_spend = amount · CPI(base) / CPI(year)`.
"""

# ╔═╡ 00000606-0000-0000-0000-000000000007
begin
	cpi = CSV.read(datadir("reference", "price_index.csv"), DataFrame)
	idx = PriceIndex(2024, Dict(Int(r.year) => Float64(r.cpi_usd) for r in eachrow(cpi)))
	DataFrame(year=cpi.year, cpi_usd=cpi.cpi_usd,
	          in_USD2024_of_1M=[round(deflate(1.0e6, Int(r.year), idx); digits=0) for r in eachrow(cpi)])
end

# ╔═╡ 00000606-0000-0000-0000-000000000008
md"### Two physical methods side by side: distance-based freight and waste treatment"

# ╔═╡ 00000606-0000-0000-0000-000000000009
begin
	lib = default_library()
	hgv = resolve_factor(lib, SOURCE_RULES["upstream_transport"]; item="hgv_diesel",
	                     date=Date(2024, 12, 31))
	waste = resolve_factor(lib, SOURCE_RULES["waste"]; item="landfill_msw",
	                       date=Date(2024, 12, 31))
	(Freight_E=emission(q"4.0e6 t·km", Quantity(hgv.value, hgv.unit)).val,
	 freight_factor=describe_factor(hgv),
	 Landfill_E=emission(q"620 t", Quantity(waste.value, waste.unit)).val,
	 waste_factor=describe_factor(waste))
end

# ╔═╡ 00000606-0000-0000-0000-000000000010
md"### From purchase-order text to an EEIO factor"

# ╔═╡ 00000606-0000-0000-0000-000000000011
begin
	po = ["stainless steel plate (2 t)", "facility management Q3", "corrugated packaging",
	      "cold-chain logistics", "IT consulting"]
	spend = [420_000.0, 96_000.0, 72_000.0, 210_000.0, 140_000.0]
	ef(s) = resolve_factor(lib, SOURCE_RULES["purchased_goods_spend"]; item=s,
	                       date=Date(2024, 12, 31)).value
	DataFrame(purchase=po, sector=sector_hint.(po), spend_USD2024=spend,
	          EF_kg_per_USD=[ef(sector_hint(p)) for p in po],
	          E_tCO₂e=[emission(Quantity(s, "USD2024"), Quantity(ef(sector_hint(p)),
	                   "kgCO₂e/USD2024")).val for (p, s) in zip(po, spend)])
end

# ╔═╡ 00000606-0000-0000-0000-000000000012
md"### The value chain of the portfolio"

# ╔═╡ 00000606-0000-0000-0000-000000000013
begin
	files = ["procurement_2024.csv", "logistics_travel_2024.csv", "waste_2024.csv",
	         "activity_erp_2024.csv", "electricity_utility_2024.csv"]
	records = ingest_files([datadir("raw", f) for f in files];
	                       entity_default="AcmeIndustrial_SA").records
	scope3 = compute_scope3(records; date=Date(2024, 12, 31), consolidation=:operational,
	                        hotspot_share=0.8)
	scope3
end

# ╔═╡ 00000606-0000-0000-0000-000000000014
begin
	bycat = sort(scope3.by_category, :E, rev=true)
	bycat.category = [first(c, 28) for c in bycat.category]
	bycat
end

# ╔═╡ 00000606-0000-0000-0000-0000000000f0
begin
	plt_cat = bar(bycat.category, bycat.E, legend=false, xrotation=20, ylabel="tCO₂e",
	              title="Scope 3 by category (sorted)", color=:steelblue)
	save_figure(plt_cat, "06_scope3_categories"; saver=savefig)
	plt_cat
end

# ╔═╡ 00000606-0000-0000-0000-000000000015
md"""
### Hotspots and the method mix

The hotspots are the categories that together carry `hotspot_share` of the total —
the refinement list. The method mix shows how much of the Scope 3 figure rests on
spend-based proxies, i.e. on estimates that must be replaced before assurance.
"""

# ╔═╡ 00000606-0000-0000-0000-000000000016
begin
	mix = combine(groupby(emissions_dataframe(scope3.rows), :method),
	              :E => sum => :E, nrow => :rows)
	mix.share = mix.E ./ sum(mix.E)
	(hotspots=scope3.hotspots, method_mix=mix)
end

# ╔═╡ 00000606-0000-0000-0000-000000000017
begin
	println("E_Scope3        : ", round(scope3.E_Scope3; digits=1), " tCO₂e")
	println("categories used : ", nrow(scope3.by_category), " of 15")
	println("hotspots        : ", join(scope3.hotspots, " · "))
	println("Notebook 06 — every category carries a method and a data-quality class, and the hotspots are named rather than assumed.")
end

