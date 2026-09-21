# ══════════════════════════════════════════════════════════════════════════════
#  report.jl — from calculated rows to a disclosed inventory
#
#  What a reported figure needs to carry with it: the consolidation boundary, the
#  data-quality index of every contributing row, the uncertainty statement, and a
#  change analysis against the prior year. `export_inventory` writes all of it —
#  rows, factors, audit ledger and a hash manifest — which is what makes the
#  submission self-proving rather than merely plausible.
# ══════════════════════════════════════════════════════════════════════════════

"Criteria of the GHG-Protocol data-quality matrix (1 = best, 5 = worst)."
const DQI_CRITERIA = ["technological representativeness", "temporal correlation",
    "geographical correlation", "completeness", "reliability of the source",
    "methodological appropriateness"]

"Weights wⱼ of the criteria (must sum to 1 conceptually; the code normalises)."
const DQI_WEIGHTS = Dict("technological representativeness" => 0.20,
    "temporal correlation" => 0.15, "geographical correlation" => 0.15,
    "completeness" => 0.15, "reliability of the source" => 0.20,
    "methodological appropriateness" => 0.15)

"""
    dqi_score(rec; reference_date, factor_region) -> NamedTuple

Data-quality index of one activity,

    DQIᵢ = Σⱼ wⱼ dᵢⱼ / Σⱼ wⱼ

with `dᵢⱼ ∈ 1…5` per criterion. The class DQ drives technological, reliability and
methodological scores; period and region add the temporal and geographical ones —
so a 2019 factor applied to a 2024 Brazilian site cannot look "perfect" simply
because it was typed in carefully.
"""
function dqi_score(rec::ActivityRecord; reference_date::Date=today(),
                   factor_region::AbstractString="GLOBAL")
    d = Dict{String,Float64}()
    d["technological representativeness"] = Float64(clamp(rec.DQ, 1, 5))
    # temporal correlation: how old is the measurement?
    age_days = abs(DateTime(rec.τ₁) - DateTime(reference_date)).value / 86_400_000
    d["temporal correlation"] = age_days <= 400 ? 1.0 : age_days <= 800 ? 2.0 : 3.0
    # geographical correlation: does the factor's region match the site?
    d["geographical correlation"] = (isempty(rec.country) || factor_region == "GLOBAL" ||
                                     factor_region == rec.country) ? 2.0 : 3.5
    # completeness: money-only or quantity-only rows are weaker than both
    d["completeness"] = (rec.amount !== nothing && rec.Q̇_canon > 0) ? 1.5 :
                        rec.Q̇_canon > 0 ? 2.0 : 4.0
    d["reliability of the source"] = Float64(clamp(rec.DQ, 1, 5))
    d["methodological appropriateness"] = rec.DQ <= 2 ? 1.0 : rec.DQ == 3 ? 2.5 : 4.0
    num = sum(DQI_WEIGHTS[c] * d[c] for c in DQI_CRITERIA)
    den = sum(DQI_WEIGHTS[c] for c in DQI_CRITERIA)
    (score=num / den, criteria=d)
end

"Interpretation of a DQI score, in words an assurance provider accepts."
function dqi_label(score::Real)
    score <= 1.5 && return "A — high quality, supplier or meter specific"
    score <= 2.5 && return "B — good, average data with documented source"
    score <= 3.5 && return "C — fair, industry average or proxy"
    "D — low, screening estimate to be refined"
end

"""
    dqi_table(rows) -> DataFrame

Emission-weighted data-quality index per source type: the table that shows *where*
the inventory's soft spots are, ranked by how many tonnes they carry.
"""
function dqi_table(rows::AbstractVector{EmissionRow})
    isempty(rows) && return DataFrame()
    by = Dict{String,Vector{EmissionRow}}()
    for r in rows
        push!(get!(by, r.source_type, EmissionRow[]), r)
    end
    out = DataFrame(source_type=String[], E=Float64[], share=Float64[], DQ=Float64[],
                    u_rel=Float64[], quality=String[])
    tot = Σᵢ(r -> r.E, rows)
    for (k, v) in sort(collect(by); by=p -> -sum(x -> x.E, p[2]))
        E = Σᵢ(x -> x.E, v)
        dq = Σᵢ(x -> x.E * x.DQ, v) / max(E, eps())
        u = Σᵢ(x -> x.E * x.u_rel, v) / max(E, eps())
        push!(out, (k, E, E / max(tot, eps()), dq, u, dqi_label(dq)))
    end
    out
end

"""
    intensity_metrics(r::InventoryResult; production_t, revenue_usd, employees) -> DataFrame

Emissions intensity — the metrics that survive growth: tCO₂e per tonne of product,
per million USD of revenue, per full-time equivalent.
"""
function intensity_metrics(r::InventoryResult; production_t::Union{Nothing,Real}=nothing,
                           revenue_usd::Union{Nothing,Real}=nothing,
                           employees::Union{Nothing,Int}=nothing)
    rows = NamedTuple{(:metric, :value, :unit),Tuple{String,Float64,String}}[]
    push!(rows, ("Scope 1 intensity", production_t === nothing ? NaN : r.E_Scope1 / production_t, "tCO₂e/t"))
    push!(rows, ("Scope 2 intensity (location)", production_t === nothing ? NaN : r.E_Scope2_loc / production_t, "tCO₂e/t"))
    push!(rows, ("Scope 3 intensity", production_t === nothing ? NaN : r.E_Scope3 / production_t, "tCO₂e/t"))
    push!(rows, ("Total intensity", production_t === nothing ? NaN : r.E_total / production_t, "tCO₂e/t"))
    push!(rows, ("Revenue intensity", revenue_usd === nothing ? NaN : r.E_total / (revenue_usd / 1e6), "tCO₂e/M\$"))
    push!(rows, ("Per employee", employees === nothing ? NaN : r.E_total / employees, "tCO₂e/FTE"))
    filter(row -> !isnan(row.value), DataFrame(rows))
end

# ── disclosure and change analysis ───────────────────────────────────────────
"Key figures of an inventory as a named tuple."
inventory_totals(r::InventoryResult) = (scope1=r.E_Scope1, scope2_loc=r.E_Scope2_loc,
    scope2_mkt=r.E_Scope2_mkt, scope3=r.E_Scope3, total=r.E_total,
    biogenic=r.scope1.memo_biogenic, λ_cov=r.scope2.λ_cov, rows=length(r.rows))

"""
    disclosure_table(r; prior, uncertainty, intensity) -> DataFrame

The disclosure table of the inventory (ESRS E1-6 / CDP shape): the three scopes,
both Scope-2 methods, the biogenic memo item, contractual coverage, every material
Scope-3 category, the uncertainty statement — and, when a prior year is passed, the
year-on-year change. No number appears without its qualification.
"""
function disclosure_table(r::InventoryResult; prior::Union{Nothing,InventoryResult}=nothing,
                          uncertainty::Union{Nothing,UncertaintySummary}=nothing,
                          intensity::Union{Nothing,DataFrame}=nothing)
    rows = NamedTuple{(:item, :value, :unit, :note),Tuple{String,Float64,String,String}}[]
    push!(rows, ("Gross Scope 1 GHG emissions", r.E_Scope1, "tCO₂e",
                 "direct emissions, " * string(r.consolidation) * " consolidation"))
    push!(rows, ("Scope 2, location based", r.E_Scope2_loc, "tCO₂e", "grid-average factors"))
    push!(rows, ("Scope 2, market based", r.E_Scope2_mkt, "tCO₂e",
                 "contractual instruments, λ_cov = $(round(100*r.scope2.λ_cov; digits=1)) %"))
    push!(rows, ("Scope 3, total value chain", r.E_Scope3, "tCO₂e", "categories 1…15, hybrid methods"))
    for cat in eachrow(r.scope3.by_category)
        push!(rows, ("   " * cat.category, cat.E, "tCO₂e",
                     "share $(round(100*cat.share; digits=1)) %, weighted DQ $(round(cat.DQ; digits=1))"))
    end
    push!(rows, ("Total GHG emissions", r.E_total, "tCO₂e", "Scope 1 + Scope 2 (location) + Scope 3"))
    push!(rows, ("Biogenic CO₂ emitted (memo)", r.scope1.memo_biogenic, "tCO₂e",
                 "excluded from the total by convention, disclosed separately"))
    if uncertainty !== nothing
        push!(rows, ("Uncertainty of the total (U₉₅)", uncertainty.U95, "tCO₂e",
                     "±$(round(100*uncertainty.u_rel; digits=1)) % at k = 2, $(uncertainty.method)"))
    end
    intensity === nothing || for i in eachrow(intensity)
        push!(rows, (i.metric, i.value, i.unit, "intensity metric"))
    end
    df = DataFrame(rows)
    if prior !== nothing
        p = inventory_totals(prior)
        lookup = Dict("Gross Scope 1 GHG emissions" => p.scope1,
                      "Scope 2, location based" => p.scope2_loc,
                      "Scope 2, market based" => p.scope2_mkt,
                      "Scope 3, total value chain" => p.scope3, "Total GHG emissions" => p.total)
        df.prior_year = [get(lookup, row.item, NaN) for row in eachrow(df)]
        df.change_pct = [(isnan(row.prior_year) || row.prior_year == 0) ? NaN :
                         100 * (row.value - row.prior_year) / row.prior_year for row in eachrow(df)]
    end
    df
end

"""
    compare_years(current, prior) -> DataFrame

Change analysis by scope: an emission change has to be explained by activity,
factors or boundary, which is why the analysis starts from the same row structure
and not from two published totals.
"""
function compare_years(current::InventoryResult, prior::InventoryResult)
    c, p = inventory_totals(current), inventory_totals(prior)
    comps = [("Scope 1", c.scope1, p.scope1), ("Scope 2 (location)", c.scope2_loc, p.scope2_loc),
             ("Scope 2 (market)", c.scope2_mkt, p.scope2_mkt), ("Scope 3", c.scope3, p.scope3),
             ("Total", c.total, p.total)]
    df = DataFrame(component=[x[1] for x in comps], current=[x[2] for x in comps],
                   prior=[x[3] for x in comps])
    df.Δ = df.current .- df.prior
    df.Δ_pct = [row.prior == 0 ? NaN : 100 * row.Δ / row.prior for row in eachrow(df)]
    df
end

# ── exports: the submission package ──────────────────────────────────────────
"""
    export_inventory(r; dir, ledger, library, sync, uncertainty, extra) -> NamedTuple

Write the submission package:

    calc_rows.csv          every emission row with factor id, share, DQ and σ
    contribution.csv       scope shares
    scope3_categories.csv  category detail with methods and DQ
    dqi.csv                data-quality index per source type
    disclosure.csv         the disclosure table
    factors.csv            the *exact* factor-library version used
    audit_ledger.json      the hash chain (when a ledger is supplied)
    manifest.json          SHA-256 of every file, the ledger head hash and the totals

The manifest is what turns a folder of CSVs into evidence: whoever receives it can
verify that nothing changed after the run.
"""
function export_inventory(r::InventoryResult; dir::AbstractString=builddir("submission"),
                          ledger::Union{Nothing,AuditLedger}=nothing,
                          library::Union{Nothing,FactorLibrary}=nothing,
                          sync::Union{Nothing,SyncState}=nothing,
                          uncertainty::Union{Nothing,UncertaintySummary}=nothing,
                          disclosure::Union{Nothing,DataFrame}=nothing,
                          extra::AbstractDict=Dict{String,String}())
    mkpath(dir)
    paths = Dict{String,String}()
    function write_csv(name, df)
        p = joinpath(dir, name)
        CSV.write(p, df)
        paths[name] = p
        p
    end
    write_csv("calc_rows.csv", inventory_rows(r))
    write_csv("contribution.csv", contribution_table(r))
    write_csv("scope3_categories.csv", r.scope3.by_category)
    write_csv("dqi.csv", dqi_table(r.rows))
    write_csv("disclosure.csv", disclosure === nothing ?
        disclosure_table(r; uncertainty=uncertainty) : disclosure)
    library === nothing || write_csv("factors.csv", factors_dataframe(library))
    if ledger !== nothing
        p = joinpath(dir, "audit_ledger.json")
        save_ledger(ledger, p)
        paths["audit_ledger.json"] = p
    end
    ok, broken = ledger === nothing ? (missing, nothing) : verify_chain(ledger)
    manifest = Dict{String,Any}(
        "generated_at" => string(now(UTC)),
        "consolidation" => string(r.consolidation),
        "totals" => Dict(string(k) => v for (k, v) in pairs(inventory_totals(r))),
        "files" => Dict(k => sha256_file(v) for (k, v) in paths),
        "ledger_head" => ledger === nothing ? "" : head_hash(ledger),
        "ledger_entries" => ledger === nothing ? 0 : length(ledger),
        "ledger_verified" => ok,
        "ledger_broken_at" => broken,
        "factor_library" => library === nothing ? "" : string(library.name, " ", library.version),
        "sync_watermark" => sync === nothing ? "" : string(sync.watermark),
        "notes" => Dict(string(k) => string(v) for (k, v) in extra))
    write(joinpath(dir, "manifest.json"), JSON.json(manifest, 2))
    paths["manifest.json"] = joinpath(dir, "manifest.json")
    (dir=dir, files=paths, manifest=manifest)
end

"Markdown summary of an inventory, used by the notebooks and the generated PDF."
function report_markdown(r::InventoryResult;
                         uncertainty::Union{Nothing,UncertaintySummary}=nothing,
                         dqi::Union{Nothing,DataFrame}=nothing,
                         prior::Union{Nothing,InventoryResult}=nothing)
    io = IOBuffer()
    t = inventory_totals(r)
    println(io, "### Inventory summary (", r.consolidation, " consolidation)\n")
    println(io, "| component | tCO₂e | share |")
    println(io, "|---|---:|---:|")
    for row in eachrow(contribution_table(r))
        @printf(io, "| %s | %.1f | %.1f %% |\n", row.component, row.E, 100 * row.share)
    end
    @printf(io, "\n- **Total**: %.1f tCO₂e over %d calculated rows\n", t.total, t.rows)
    @printf(io, "- **Biogenic CO₂ (memo)**: %.1f tCO₂e\n", t.biogenic)
    @printf(io, "- **Contractual instrument coverage**: %.1f %% of purchased electricity\n", 100 * t.λ_cov)
    uncertainty === nothing || println(io, "- **Uncertainty**: ", uncertainty_statement(uncertainty))
    if dqi !== nothing && nrow(dqi) > 0
        worst = first(eachrow(sort(dqi, :DQ, rev=true)))
        @printf(io, "- **Weakest data quality**: %s (DQ %.1f, %.1f %% of emissions)\n",
                worst.source_type, worst.DQ, 100 * worst.share)
    end
    if prior !== nothing
        println(io, "\nChange versus the prior year:\n")
        for row in eachrow(compare_years(r, prior))
            isnan(row.Δ_pct) && continue
            @printf(io, "- %s: %+.1f tCO₂e (%+.1f %%)\n", row.component, row.Δ, row.Δ_pct)
        end
    end
    String(take!(io))
end
