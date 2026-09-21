### A Pluto.jl notebook ###
# v1.0.3
#
# 10 — Rollup, uncertainty, data quality, disclosure and the submission package.

using Markdown
using InteractiveUtils

# ╔═╡ 00001010-0000-0000-0000-000000000001
begin
	import Pkg
	Pkg.activate(normpath(joinpath(@__DIR__, "..")))
end

# ╔═╡ 00001010-0000-0000-0000-000000000002
begin
	using CarbonAccounting, CSV, DataFrames, Dates, Statistics, Printf
	using Plots
	default(fmt=:svg, legend=:topleft, size=(760, 420), dpi=120)
	datadir(parts...) = CarbonAccounting.datadir(parts...)
end

# ╔═╡ 00001010-0000-0000-0000-000000000003
md"""
# 10 · The inventory: rollup, uncertainty, data quality and disclosure

Three things leave this notebook: a **consolidated total**, an **uncertainty
statement**, and a **submission package** whose integrity can be verified by whoever
receives it.

    E_total = E_Scope1 + E_Scope2(location) + E_Scope3
    u_c²    = Σ_g [ Σ_{i∈g}(Eᵢ u_act,ᵢ)² + (A_g u_fac,g)² ] + 2ρ Σ_{g<h} A_g A_h u_fac,g u_fac,h
    DQIᵢ    = Σⱼ wⱼ dᵢⱼ / Σⱼ wⱼ
"""

# ╔═╡ 00001010-0000-0000-0000-000000000004
begin
	ledger = AuditLedger(hmac_key=Vector{UInt8}(codeunits("notebook-10")))
	analyst = Actor("u001", "A. Analyst", :analyst)
	files = ["activity_erp_2024.csv", "electricity_utility_2024.csv", "procurement_2024.csv",
	         "logistics_travel_2024.csv", "waste_2024.csv"]
	inventory = let
		recs = ingest_files([datadir("raw", f) for f in files];
		                    entity_default="AcmeIndustrial_SA", ledger=ledger,
		                    actor=analyst).records
		compute_inventory(recs; consolidation=:operational, date=Date(2024, 12, 31),
		                  ledger=ledger, actor=analyst)
	end
	inventory
end

# ╔═╡ 00001010-0000-0000-0000-000000000005
md"""
### Consolidation ©ᵢ changes *which* activities enter the boundary

Equity share weights every activity by ownership; the control approaches take all of
it. The comparison is therefore a boundary disclosure, not a sensitivity.
"""

# ╔═╡ 00001010-0000-0000-0000-000000000006
begin
	recs = ingest_files([datadir("raw", f) for f in files];
	                    entity_default="AcmeIndustrial_SA").records
	# the equity shares ©ᵢ come from the master data, keyed by site
	master_eq = CSV.read(datadir("reference", "master_data.csv"), DataFrame)
	shares = Dict(String(r.site) => Float64(r.equity_share) for r in eachrow(master_eq))
	eq = compute_inventory(recs; consolidation=:equity, shares=shares, date=Date(2024, 12, 31))
	fs = compute_inventory(recs; consolidation=:financial, date=Date(2024, 12, 31))
	DataFrame(approach=["operational control", "financial control", "equity share"],
	          E_Scope1=[inventory.E_Scope1, fs.E_Scope1, eq.E_Scope1],
	          E_Scope2_loc=[inventory.E_Scope2_loc, fs.E_Scope2_loc, eq.E_Scope2_loc],
	          E_Scope3=[inventory.E_Scope3, fs.E_Scope3, eq.E_Scope3],
	          E_total=[inventory.E_total, fs.E_total, eq.E_total])
end

# ╔═╡ 00001010-0000-0000-0000-000000000007
md"""
### Two routes to the uncertainty of the total

The **GUM** route propagates the sensitivity coefficients analytically, keeping the
activity part independent and the factor part correlated inside each factor group.
The **Monte Carlo** route draws each row's activity and factor and recomputes the
total — which also captures non-linearity and asymmetry. Agreement between the two is
the evidence that the model is nearly linear; disagreement is a finding.

Both routes run over `total_rows(inventory)`, i.e. Scope 1, the *location-based*
Scope 2 rows and Scope 3 — the rows that sum to `E_total`. The market-based Scope 2
rows sit in the same table but are a parallel report, not an addition.
"""

# ╔═╡ 00001010-0000-0000-0000-000000000008
begin
	rows_total = total_rows(inventory)
	gum = gum_uncertainty(rows_total)
	mc = monte_carlo_uncertainty(rows_total; n_MC=5_000, seed=42)
	(rows_in_total=length(rows_total), of_all=length(inventory.rows), gum, mc)
end

# ╔═╡ 00001010-0000-0000-0000-000000000009
uncertainty_table(gum, mc)

# ╔═╡ 00001010-0000-0000-0000-000000000010
begin
	plt = histogram(mc.samples, bins=60, legend=false, color=:steelblue, alpha=0.7,
	                xlabel="tCO₂e", ylabel="Monte-Carlo trials",
	                title="Inventory uncertainty (n = $(length(mc.samples)) trials)")
	vline!(plt, [mc.E], color=:black, ls=:dash, label="central")
	vline!(plt, [mc.lo95, mc.hi95], color=:red, ls=:dot, label="95 % interval")
	save_figure(plt, "10_uncertainty_histogram"; saver=savefig)
	plt
end

# ╔═╡ 00001010-0000-0000-0000-000000000011
md"""
### Where the quality sits, and the intensity metrics that survive growth
"""

# ╔═╡ 00001010-0000-0000-0000-000000000012
begin
	master = CSV.read(datadir("reference", "master_data.csv"), DataFrame)
	production = sum(master.production_t)
	revenue = sum(master.revenue_usd)
	employees = sum(master.employees)
	(dqi=first(sort(dqi_table(inventory.rows), :DQ, rev=true), 5),
	 intensity=intensity_metrics(inventory; production_t=production, revenue_usd=revenue,
	                             employees=employees),
	 denominators=(production_t=production, revenue_USD=revenue, employees=employees))
end

# ╔═╡ 00001010-0000-0000-0000-000000000013
md"""
### Year-on-year change needs a prior year computed the same way

The prior period is built by re-ingesting the same export with volumes 7 % lower —
through the identical pipeline and with the **same factor set** (the same factor
date), so a change is attributable to activity instead of to factor churn.
"""

# ╔═╡ 00001010-0000-0000-0000-000000000014
begin
	prior_dir = CarbonAccounting.builddir("notebook10", "prior")
	mkpath(prior_dir)
	src = CSV.read(datadir("raw", "activity_erp_2024.csv"), DataFrame; delim=';',
	               types=String)
	scaled = copy(src)
	scaled[!, "Verbrauch"] = [string(round(parse_number(x) * 0.93; digits=1)) for x in src.Verbrauch]
	CSV.write(joinpath(prior_dir, "activity_erp_prior.csv"), scaled; delim=';')
	prior_recs = vcat(ingest_files([joinpath(prior_dir, "activity_erp_prior.csv")];
	                               entity_default="AcmeIndustrial_SA").records,
	                  [r for r in recs if !occursin("activity_erp", r.source)])
	prior = compute_inventory(prior_recs; consolidation=:operational, date=Date(2024, 12, 31))
	compare_years(inventory, prior)
end

# ╔═╡ 00001010-0000-0000-0000-000000000015
md"### The disclosure table — every figure with its qualification"

# ╔═╡ 00001010-0000-0000-0000-000000000016
begin
	disp = disclosure_table(inventory; prior=prior, uncertainty=mc,
	                        intensity=intensity_metrics(inventory; production_t=production,
	                                                    revenue_usd=revenue, employees=employees))
	first(disp, 12)
end

# ╔═╡ 00001010-0000-0000-0000-000000000017
md"""
### The submission package

Rows, contribution, Scope-3 categories, DQI, the disclosure table, the exact factor
library used, the audit ledger — and a **manifest** with the SHA-256 of every file
and the ledger head hash, so the recipient can verify that nothing changed after the
run.
"""

# ╔═╡ 00001010-0000-0000-0000-000000000018
begin
	package = export_inventory(inventory; dir=CarbonAccounting.builddir("submission"),
	                           ledger=ledger, library=default_library(), uncertainty=mc,
	                           extra=Dict("reporting_year" => "2024",
	                                      "consolidation" => "operational control"))
	(files=sort(collect(keys(package.files))),
	 manifest_entries=sort(collect(keys(package.manifest))),
	 ledger_verified=package.manifest["ledger_verified"],
	 factor_library=package.manifest["factor_library"])
end

# ╔═╡ 00001010-0000-0000-0000-000000000019
Markdown.parse(report_markdown(inventory; uncertainty=mc,
                               dqi=dqi_table(inventory.rows), prior=prior))

# ╔═╡ 00001010-0000-0000-0000-000000000020
begin
	t = inventory_totals(inventory)
	println("E_Scope1        : ", round(t.scope1; digits=1), " tCO₂e")
	println("E_Scope2 loc/mkt: ", round(t.scope2_loc; digits=1), " / ",
	        round(t.scope2_mkt; digits=1), " tCO₂e")
	println("E_Scope3        : ", round(t.scope3; digits=1), " tCO₂e")
	println("E_total         : ", round(t.total; digits=1), " tCO₂e")
	println("uncertainty     : ", uncertainty_statement(mc))
	println("chain           : ", chain_statement(ledger))
	println("Notebook 10 — the inventory is consolidated, uncertainty-stated, quality-rated, disclosable and packaged with a verifiable manifest.")
end

