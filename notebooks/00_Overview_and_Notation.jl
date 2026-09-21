### A Pluto.jl notebook ###
# v1.0.3
#
# 00 — Overview, architecture and the symbolic notation.
# Code cells are wrapped in `begin … end` so that each one is a single expression
# (the portable cell style, independent of the Pluto version).

using Markdown
using InteractiveUtils

# ╔═╡ 00000001-0000-0000-0000-000000000001
begin
	# this repository pins ONE environment for library, notebooks and build scripts
	import Pkg
	Pkg.activate(normpath(joinpath(@__DIR__, "..")))
end

# ╔═╡ 00000001-0000-0000-0000-000000000002
begin
	using CarbonAccounting, CSV, DataFrames, Dates, Statistics, Printf
	using Plots
	default(fmt=:svg, legend=:topleft, size=(760, 420), dpi=120)
	datadir(parts...) = CarbonAccounting.datadir(parts...)
end

# ╔═╡ 00000001-0000-0000-0000-000000000003
md"""
# Carbon accounting in Julia — platform overview

**Scope 1, 2 and 3 · audit trails and security · parsing and normalisation ·
calculation engines and emission-factor libraries · integration and connectivity ·
a neural soft sensor for the annual analyzer campaign · anomaly detection.**

The repository is a Julia package (`src/CarbonAccounting.jl`) plus eleven Pluto
notebooks that walk through it, plus a build pipeline that turns the notebooks into
the PDF you are reading.

| module | responsibility | notebook |
|---|---|---|
| `notation.jl` | symbolic legend, GWP sets, `Σᵢ` | **00** |
| `ingest.jl` | parsing, normalisation, quarantine, lineage | **01** |
| `connect.jl` | connectors, incremental sync, idempotency | **02** |
| `factors.jl` | versioned emission-factor library | **03** |
| `engines.jl` | Scope 1 / 2 / 3 calculation engines | **04, 05, 06** |
| `softsensor.jl` | neural soft sensor + campaign design | **07** |
| `anomaly.jl` | EWMA, CUSUM, T², SPE, autoencoder | **08** |
| `security.jl` | RBAC, hash-chained audit trail, sealing | **09** |
| `uncertainty.jl`, `report.jl` | GUM + Monte Carlo, DQI, disclosure, exports | **10** |

Three design rules run through every file:

1. **The code carries the equation.** Every quantity keeps its mathematical symbol
   (`Q̇ᵢ`, `EFᵢ`, `σ̂`, `W⁽ˡ⁾`, `r̂ₜ`, `hₙ`), so a cell can be read next to the formula
   it implements.
2. **A number is never reported without its qualification.** Provenance (source
   hash), data-quality class DQ, factor version and uncertainty travel with each row.
3. **Nothing is silently repaired.** Rows that cannot be interpreted are quarantined
   with a reason; every state change lands in a hash-chained ledger.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000004
md"""
## 1. The notation legend is part of the code

`NOTATION` is a table of `(symbol, LaTeX, meaning, unit, area)`; `check_notation()`
asserts that every symbol in it is a legal Julia identifier, so documentation and
source cannot drift apart.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000005
length(NOTATION), NOTATION_AREAS, check_notation()

# ╔═╡ 00000001-0000-0000-0000-000000000006
Markdown.parse(notation_markdown(area="activity"))

# ╔═╡ 00000001-0000-0000-0000-000000000007
Markdown.parse(notation_markdown(area="softsensor"))

# ╔═╡ 00000001-0000-0000-0000-000000000008
md"""
## 2. Units are types, so `E = Q̇ · EF` cannot be wrong by accident

A factor is converted into the unit the activity is measured in — and refused when
the *dimension* does not match:

| activity | factor | result |
|---|---|---|
| `1250 MWh` | `0.42 kgCO₂e/kWh` | 525 tCO₂e |
| `1250 MWh` | `0.42 kgCO₂e/GJ` | 1.89 tCO₂e — MWh → GJ, faithfully converted |
| `1250 MWh` | `0.42 kgCO₂e/t` | **`DimensionMismatch`** — energy vs mass |

Canonical units: mass `t`, CO₂e `tCO₂e`, energy `GJ`, volume `m³` (normal
conditions), distance `km`, transport work `t·km`, money `USD2024`, temperature `K`.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000009
begin
	inputs = ["1,250 MWh", "1,250 kWh", "3,412 Btu", "1 toe", "2,204.6 lb", "1,000 scf"]
	q = parse_quantity.(inputs)
	DataFrame(as_written=inputs, value_canonical=val_canon.(q),
	          canonical_unit=[CANONICAL[dimension(x)] for x in q])
end

# ╔═╡ 00000001-0000-0000-0000-000000000010
begin
	Q̇ᵢ = q"1250 MWh"                       # the activity as the utility reports it
	EFᵢ = q"0.42 kgCO₂e/kWh"               # the factor as the library holds it
	(Eᵢ = emission(Q̇ᵢ, EFᵢ),
	 same_energy_in_GJ = emission(q"1250 MWh", q"0.42 kgCO₂e/GJ"),
	 refused = try
		 emission(q"1250 MWh", q"0.42 kgCO₂e/t")     # energy × mass factor: refused
	 catch e
		 sprint(showerror, e)
	 end)
end

# ╔═╡ 00000001-0000-0000-0000-000000000011
md"""
## 3. The GWP conversion

`E_CO₂e = E_CO₂ + GWP_CH₄·E_CH₄ + GWP_N₂O·E_N₂O`. The set used is part of the
disclosure: AR6 separates **fossil** from **biogenic** methane (29.8 vs 27.0 over
100 years), so a biogenic slip valued with the fossil factor is 10 % too high.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000012
begin
	t = 1.0                                    # one tonne of each gas
	DataFrame(gas=["CH₄ fossil", "CH₄ biogenic", "N₂O", "CO₂"],
	          AR4=[co2e(; CH₄=t, gwp=GWP_AR4), co2e(; CH₄=t, gwp=GWP_AR4, fossil=false),
	               co2e(; N₂O=t, gwp=GWP_AR4), t],
	          AR5=[co2e(; CH₄=t, gwp=GWP_AR5), co2e(; CH₄=t, gwp=GWP_AR5, fossil=false),
	               co2e(; N₂O=t, gwp=GWP_AR5), t],
	          AR6=[co2e(; CH₄=t), co2e(; CH₄=t, fossil=false), co2e(; N₂O=t), t])
end

# ╔═╡ 00000001-0000-0000-0000-000000000013
md"""
## 4. The platform end to end

The remaining cells run the whole chain on the repository's synthetic portfolio —
six sites in five countries, one reporting year — in a few seconds, so that the
following notebooks can go deep on one link at a time:

    raw exports → ingest → normalise → factors → engines → uncertainty → disclosure
                                    ↘ soft sensor ↘ anomaly detection ↘ audit trail ↗
"""

# ╔═╡ 00000001-0000-0000-0000-000000000014
begin
	paths = [datadir("raw", f) for f in
	         ["activity_erp_2024.csv", "electricity_utility_2024.csv", "procurement_2024.csv",
	          "logistics_travel_2024.csv", "waste_2024.csv"]]
	records = ingest_files(paths; entity_default="AcmeIndustrial_SA").records
	inventory = compute_inventory(records; consolidation=:operational, date=Date(2024, 12, 31))
	inventory
end

# ╔═╡ 00000001-0000-0000-0000-000000000015
contribution_table(inventory)

# ╔═╡ 00000001-0000-0000-0000-000000000016
begin
	contrib = contribution_table(inventory)
	plt = bar(contrib.component, contrib.E,
	          title="Greenhouse-gas inventory 2024 (operational control)",
	          ylabel="tCO₂e", xrotation=10, legend=false,
	          color=[:steelblue :seagreen :darkseagreen :darkorange])
	save_figure(plt, "00_contribution"; saver=savefig)
	plt
end

# ╔═╡ 00000001-0000-0000-0000-000000000017
md"""
## 5. How to read the rest

* every notebook activates the same environment and reads the same files, so each can
  be run on its own;
* cells keep the symbols of the equations: where a cell reads `E_Scope1 = Σᵢ(…)`, the
  expression between `Σᵢ` and the closing parenthesis is the summand of the published
  formula;
* the last cell of each notebook states what was **verified**, not merely what was
  computed — the same sentence appears in the PDF printout.

The PDF in `docs/` is produced by `julia --project=. scripts/build_pdf.jl`, which runs
these notebooks headlessly, renders the cell outputs to HTML and prints that to PDF
with headless Chrome.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000018
md"""
---
*Notebook 00 of the Carbon Accounting platform · full documentation in `README.md`.*
"""

# ╔═╡ 00000001-0000-0000-0000-000000000019
begin
	# last cell: state what was *verified*, not merely what was computed
	println("notation symbols checked : ", length(NOTATION), " (invalid: ", check_notation(), ")")
	println("activity records         : ", length(records))
	println("inventory total          : ", round(inventory.E_total; digits=1), " tCO₂e over ",
	        length(inventory.rows), " calculated rows")
	println("Notebook 00 — the legend and the source code are consistent, and the platform runs end to end.")
end

