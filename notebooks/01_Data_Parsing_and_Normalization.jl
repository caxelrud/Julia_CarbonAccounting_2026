### A Pluto.jl notebook ###
# v1.0.3
#
# 01 — Data parsing and normalisation: from messy exports to typed activity data.

using Markdown
using InteractiveUtils

# ╔═╡ 00000101-0000-0000-0000-000000000001
begin
	import Pkg
	Pkg.activate(normpath(joinpath(@__DIR__, "..")))
end

# ╔═╡ 00000101-0000-0000-0000-000000000002
begin
	using CarbonAccounting, CSV, DataFrames, Dates, Statistics, Printf
	using Plots
	default(fmt=:svg, legend=:topleft, size=(760, 420), dpi=120)
	datadir(parts...) = CarbonAccounting.datadir(parts...)
end

# ╔═╡ 00000101-0000-0000-0000-000000000003
md"""
# 01 · Data parsing and normalisation

Raw activity data arrives from meters, ERP extracts, fuel invoices and utility
portals: mixed delimiters, units glued to the value (`"1,234.5 MWh"`), decimal
commas, several date conventions, multi-language headers, duplicated rows.

`src/ingest.jl` turns that into typed `ActivityRecord`s carrying a **canonical
unit** per dimension (`t`, `GJ`, `m³`, `kWh`, `USD2024`, `t·km`), an explicit period
𝒫 = [τ₀, τ₁], a **data-quality class** DQ ∈ {1,2,3,4} and full **lineage** — and it
*quarantines* every row it cannot interpret, with the reason, instead of silently
dropping or coercing it.
"""

# ╔═╡ 00000101-0000-0000-0000-000000000004
md"### The raw export, exactly as it arrived — semicolons, decimal commas, German headers"

# ╔═╡ 00000101-0000-0000-0000-000000000005
begin
	f = datadir("raw", "activity_erp_2024.csv")
	println.(first(eachline(f), 8))
	f
end

# ╔═╡ 00000101-0000-0000-0000-000000000006
md"""
### Scalar parsers

`parse_number` accepts `1.234,5` as well as `1,234.5`; `parse_period` resolves
`2024-Q1`, `2024-03`, `2024` and `01.03.2024-31.03.2024` into 𝒫 = [τ₀, τ₁]; the
`dayfirst` flag resolves `01/02/2024` explicitly instead of guessing.
"""

# ╔═╡ 00000101-0000-0000-0000-000000000007
begin
	nums = ["1.234,5", "1,234.5", "(1 250,75)", "12,5", "3.141,59"]
	dates = ["2024-03-31", "31/03/2024", "31.03.2024", "2024-03"]
	(numeric=[parse_number(s) for s in nums],
	 dates=[parse_date(s) for s in dates],
	 periods=[string(parse_period(s)) for s in
	          ["2024-Q1", "2024-03", "2024", "01.03.2024-31.03.2024"]])
end

# ╔═╡ 00000101-0000-0000-0000-000000000008
md"""
### Ingest the five exports

Every file is hashed (SHA-256) and — because a ledger and an actor are passed — the
ingestion itself is written to the audit trail, so the *provenance* of every figure
is provable rather than asserted.
"""

# ╔═╡ 00000101-0000-0000-0000-000000000009
begin
	ledger = AuditLedger(hmac_key=Vector{UInt8}(codeunits("notebook-01")))
	analyst = Actor("u001", "A. Analyst", :analyst)
	files = ["activity_erp_2024.csv", "electricity_utility_2024.csv", "procurement_2024.csv",
	         "logistics_travel_2024.csv", "waste_2024.csv"]
	report = ingest_files([datadir("raw", f) for f in files];
	                      entity_default="AcmeIndustrial_SA", ledger=ledger, actor=analyst,
	                      context="notebook 01 — ingest run")
	report
end

# ╔═╡ 00000101-0000-0000-0000-000000000010
md"""
### What was quarantined, and why

The quarantine is the audit-friendly alternative to silent repair: the rows of the
demo that cannot be interpreted appear here with their reason.
"""

# ╔═╡ 00000101-0000-0000-0000-000000000011
begin
	q = quarantine_dataframe(report.quarantined)
	isempty(q) ? q : combine(groupby(q, :reason), nrow => :rows)
end

# ╔═╡ 00000101-0000-0000-0000-000000000012
begin
	records = report.records
	df = records_dataframe(records)
	select(df, :rec_id, :site, :source_type, :item, :τ₀, :value, :unit, :Q̇_canon,
	       :unit_canon, :DQ, :source, :source_hash)
end

# ╔═╡ 00000101-0000-0000-0000-000000000013
md"""
### Normalisation is a *dimensional* step, not a cosmetic one

`1.000 L` of diesel, `4,200 kg` of the same fuel and `57 GJ` of heat all land in the
canonical unit of their dimension. The engines downstream only ever see canonical
units, which is why a unit mix-up cannot survive into the total.
"""

# ╔═╡ 00000101-0000-0000-0000-000000000014
combine(groupby(records_dataframe(records), [:scope, :unit, :unit_canon]),
        nrow => :records, :Q̇_canon => sum => :activity)

# ╔═╡ 00000101-0000-0000-0000-000000000015
md"""
### Reporting controls before any calculation

`validate_records` applies the checks an auditor asks for first: non-positive
activity, missing entity, periods outside the reporting year, proxy-quality rows,
implausible magnitudes. `:error` rows must be corrected; `:info` rows are those to
refine with supplier data.
"""

# ╔═╡ 00000101-0000-0000-0000-000000000016
begin
	issues = validate_records(records; year=2024)
	(by_severity=combine(groupby(DataFrame(issues), :severity), nrow => :n),
	 examples=[(i.rec_id, i.severity, i.message) for i in first(issues, 4)])
end

# ╔═╡ 00000101-0000-0000-0000-000000000017
md"""
### Data quality weighted by emissions, and the lineage of every file

DQ 1 is supplier- or meter-specific, DQ 4 a proxy or spend-based estimate. The
weighted table shows where the inventory's soft spots are, ranked by the tonnes they
carry — and `report.files` holds the SHA-256 of every file ingested.
"""

# ╔═╡ 00000101-0000-0000-0000-000000000018
begin
	inventory = compute_inventory(records; consolidation=:operational, date=Date(2024, 12, 31))
	dqi_table(inventory.rows)
end

# ╔═╡ 00000101-0000-0000-0000-000000000019
DataFrame(file=basename.([string(f["path"]) for f in report.files]),
          rows=[f["rows"] for f in report.files],
          delimiter=[f["delim"] for f in report.files],
          sha256=[first(string(f["sha256"]), 16) for f in report.files])

# ╔═╡ 00000101-0000-0000-0000-000000000020
begin
	println("files ingested      : ", length(report.files))
	println("records accepted    : ", length(records))
	println("rows quarantined    : ", length(report.quarantined), " (reasons above)")
	println("validation findings : ", length(issues))
	println("audit trail         : ", chain_statement(ledger))
	println("Notebook 01 — every row is typed, dimension-checked, quality-tagged and traceable.")
end

