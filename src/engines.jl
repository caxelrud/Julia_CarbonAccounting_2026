# ══════════════════════════════════════════════════════════════════════════════
#  engines.jl — the calculation engines
#
#  Every engine returns *rows*, never a bare number: one row is one
#  activity × factor product with its provenance,
#
#      Eᵢ = Q̇ᵢ · EFᵢ · ©ᵢ            (©ᵢ = consolidation share)
#
#  so any total can be decomposed back to the source file, the factor version
#  and the data-quality class. Scope 1 (E_stat, E_mob, E_fug, E_flare, E_proc),
#  Scope 2 (E_loc and E_mkt, dual reporting) and Scope 3 (categories 1…15,
#  hybrid data hierarchy with screening) are all built from the same row type.
# ══════════════════════════════════════════════════════════════════════════════

"""
    EmissionRow

One calculated emission with full provenance. `E` is always in tCO₂e.
"""
Base.@kwdef struct EmissionRow
    rec_id::String = ""
    entity_id::String = ""
    site_id::String = ""
    source_type::String = ""
    scope::Symbol = :scope1
    cat::Union{Nothing,Int} = nothing
    item::String = ""
    method::String = ""                 # supplier-specific | average data | spend based | …
    Q̇::Float64 = 0.0                     # activity in its canonical unit
    unit::String = ""
    ef_id::String = ""
    ef_value::Float64 = 0.0
    ef_unit::String = ""
    library::String = ""
    version::String = ""
    ef_source::String = ""
    ©::Float64 = 1.0                     # consolidation share
    E::Float64 = 0.0                     # tCO₂e (= Q̇ · EF · ©)
    DQ::Int = 3
    u_rel::Float64 = 0.0                 # combined relative standard uncertainty
    u_act::Float64 = 0.0                 # of the activity alone
    u_fac::Float64 = 0.0                 # of the factor alone (correlated within a factor group)
end

"One-line rendering of a calculated row."
function Base.show(io::IO, r::EmissionRow)
    @printf(io, "%s │ %-22s │ %-24s │ %9.3f %s │ %-16s │ %10.4f tCO₂e │ DQ%d",
            r.rec_id, r.source_type, r.item, r.Q̇, r.unit, r.method, r.E, r.DQ)
end

"Emissions rows as a DataFrame."
emissions_dataframe(rows::AbstractVector{EmissionRow}) = DataFrame(
    rec_id=[r.rec_id for r in rows], entity=[r.entity_id for r in rows], site=[r.site_id for r in rows],
    source_type=[r.source_type for r in rows], scope=[string(r.scope) for r in rows],
    cat=[r.cat === nothing ? missing : r.cat for r in rows], item=[r.item for r in rows],
    method=[r.method for r in rows], Q̇=[r.Q̇ for r in rows], unit=[r.unit for r in rows],
    ef_id=[r.ef_id for r in rows], ef_value=[r.ef_value for r in rows], ef_unit=[r.ef_unit for r in rows],
    library=[r.library for r in rows], version=[r.version for r in rows],
    share=[r.© for r in rows], E=[r.E for r in rows], DQ=[r.DQ for r in rows], u_rel=[r.u_rel for r in rows])

"""
    Scope1Result

Direct emissions. `by_source` holds `E_stat`, `E_mob`, `E_fug`, `E_flare`,
`E_proc`; `memo_biogenic` records the biogenic CO₂ excluded from the scope-1
total that still has to be disclosed as a memo item.
"""
struct Scope1Result
    rows::Vector{EmissionRow}
    by_source::Dict{String,Float64}
    E_Scope1::Float64
    memo_biogenic::Float64
end

"""
    Scope2Result

Purchased energy. `E_loc` is the location-based figure, `E_mkt` the market-based
one — the GHG Protocol requires **both** — and `λ_cov` the share of the purchased
energy covered by contractual instruments.
"""
struct Scope2Result
    rows::Vector{EmissionRow}
    E_loc::Float64
    E_mkt::Float64
    C_elec::Float64
    λ_cov::Float64
    by_site::DataFrame
end

"""
    Scope3Result

Value-chain emissions: totals per category 1…15, the row detail, and the screening
result (Pareto hotspots that have to be refined with better data).
"""
struct Scope3Result
    rows::Vector{EmissionRow}
    by_category::DataFrame
    E_Scope3::Float64
    hotspots::Vector{String}
end

"""
    InventoryResult

The consolidated inventory: Scope 1, Scope 2 (both methods), Scope 3, the grand
total and the combined detail.
"""
struct InventoryResult
    scope1::Scope1Result
    scope2::Scope2Result
    scope3::Scope3Result
    consolidation::Symbol
    E_Scope1::Float64
    E_Scope2_loc::Float64
    E_Scope2_mkt::Float64
    E_Scope3::Float64
    E_total::Float64
    rows::Vector{EmissionRow}
end

function Base.show(io::IO, r::InventoryResult)
    @printf(io, "InventoryResult [%s]\n", r.consolidation)
    @printf(io, "  E_Scope1     = %12.3f tCO₂e\n", r.E_Scope1)
    @printf(io, "  E_Scope2 loc = %12.3f tCO₂e\n", r.E_Scope2_loc)
    @printf(io, "  E_Scope2 mkt = %12.3f tCO₂e\n", r.E_Scope2_mkt)
    @printf(io, "  E_Scope3     = %12.3f tCO₂e\n", r.E_Scope3)
    @printf(io, "  E_total      = %12.3f tCO₂e  (%d rows)\n", r.E_total, length(r.rows))
end

# ── consolidation shares ©ᵢ ──────────────────────────────────────────────────
"""
    financial_control(rec) = 1
    operational_control(rec) = 1
    equity_share(rec; shares)

Consolidation share ©ᵢ of one activity: full share under the control approaches,
ownership share under the equity approach (read from `meta["equity_share"]` or a
`shares` dictionary keyed by entity).
"""
financial_control(::ActivityRecord) = 1.0
operational_control(::ActivityRecord) = 1.0
function equity_share(rec::ActivityRecord; shares::AbstractDict=Dict{String,Float64}())
    haskey(shares, rec.entity_id) && return Float64(shares[rec.entity_id])
    haskey(shares, rec.site_id) && return Float64(shares[rec.site_id])
    s = strip(get(rec.meta, "equity_share", ""))
    isempty(s) ? 1.0 : parse(Float64, s)
end

const CONSOLIDATION_METHODS = Dict(
    :financial => "financial control (©ᵢ = 1 for controlled entities)",
    :operational => "operational control (©ᵢ = 1 for operated assets)",
    :equity => "equity share (©ᵢ = ownership share)")

"Consolidation share ©ᵢ of a record under the chosen approach."
function consolidate(method::Symbol, rec::ActivityRecord; shares::AbstractDict=Dict{String,Float64}())
    method == :equity && return equity_share(rec; shares=shares)
    method == :financial && return financial_control(rec)
    method == :operational && return operational_control(rec)
    throw(ArgumentError("unknown consolidation approach «$method»"))
end

"Build one `EmissionRow` from an activity and a factor (the generic E = Q̇·EF)."
function emission_row(rec::ActivityRecord, f::EFRecord; E::Float64, method::AbstractString,
                      ©::Float64=1.0, u_act::Float64=u_activity(rec), u_fac::Float64=f.uncertainty)
    EmissionRow(rec_id=rec.rec_id, entity_id=rec.entity_id, site_id=rec.site_id,
        source_type=rec.source_type, scope=rec.scope, cat=rec.cat, item=rec.item,
        method=String(method), Q̇=rec.Q̇_canon, unit=rec.unit_canon, ef_id=f.ef_id,
        ef_value=f.value, ef_unit=f.unit, library=f.library, version=f.version,
        ef_source=f.source, ©=©, E=E, DQ=min(rec.DQ, f.DQ), u_rel=row_u_rel(u_act, u_fac),
        u_act=u_act, u_fac=u_fac)
end

"Combine the relative uncertainties of activity and factor in quadrature (1σ)."
row_u_rel(u_activity::Real, u_factor::Real) = sqrt(max(u_activity, 0.0)^2 + max(u_factor, 0.0)^2)


# ══════════════════════ Scope 1 — direct emissions ═══════════════════════════
"""
    combustion_emissions(Q̇; EF_CO₂, EF_CH₄, EF_N₂O, OF, η_b, gwp) -> NamedTuple

Combustion, gas by gas (IPCC method 1):

    E_CO₂ = Q̇/η_b · EF_CO₂ · OF     E_CH₄ = Q̇/η_b · EF_CH₄     E_N₂O = Q̇/η_b · EF_N₂O
    E     = E_CO₂ + GWP_CH₄·E_CH₄ + GWP_N₂O·E_N₂O

Factors in kg per GJ, `η_b` the boiler efficiency, OF the oxidation factor; the
result is in **tCO₂e**.
"""
function combustion_emissions(Q̇::Real; EF_CO₂::Real, EF_CH₄::Real=0.0, EF_N₂O::Real=0.0,
                              OF::Real=1.0, η_b::Real=1.0, gwp::GWPSet=GWP_AR6, fossil::Bool=true)
    Q̇_in = Q̇ / η_b                                  # fuel-energy input
    m_CO₂ = Q̇_in * EF_CO₂ * OF
    m_CH₄ = Q̇_in * EF_CH₄
    m_N₂O = Q̇_in * EF_N₂O
    E = co2e(; CO₂=m_CO₂, CH₄=m_CH₄, N₂O=m_N₂O, gwp=gwp, fossil=fossil) / 1000
    (E=E, E_CO₂=m_CO₂ / 1000, E_CH₄=m_CH₄ / 1000, E_N₂O=m_N₂O / 1000, Q̇_in=Q̇_in)
end

"""
    flare_emissions(V̇; EF_flare, DRE, η_comb, x_CH₄, gwp) -> NamedTuple

Flaring is *not* a single multiplication: only the fraction `φ = DRE · η_comb` of
the gas is destroyed, and the slip is a methane emission of its own,

    E_combusted = V̇ · EF_flare · φ
    m_CH₄,slip  = V̇ · ρ_CH₄ · x_CH₄ · (1 − φ)
    E           = E_combusted + GWP_CH₄ · m_CH₄,slip
"""
function flare_emissions(V̇::Real; EF_flare::Real, DRE::Real=0.98, η_comb::Real=0.995,
                         x_CH₄::Real=0.80, ρ_CH₄_::Real=ρ_CH₄, gwp::GWPSet=GWP_AR6)
    φ = DRE * η_comb
    E_combusted = V̇ * EF_flare * φ / 1000
    m_slip = V̇ * ρ_CH₄_ * x_CH₄ * (1 - φ)
    E_slip = co2e(; CH₄=m_slip, gwp=gwp) / 1000
    (E=E_combusted + E_slip, E_combusted=E_combusted, E_slip=E_slip, φ=φ, m_CH₄_slip=m_slip)
end

"""
    fugitive_refrigerant(; C_start, C_purchased, C_end, C_recovered, C_disposed, GWP₁₀₀)

Refrigerant mass balance (IPCC method 1):

    m_lost = C_start + C_purchased − C_end − C_recovered − C_disposed,     E = m_lost · GWP₁₀₀

charges in kg, `GWP₁₀₀` from the library (R410A = 2088, R134a = 1430, …).
"""
function fugitive_refrigerant(; C_start::Real, C_purchased::Real, C_end::Real,
                              C_recovered::Real=0.0, C_disposed::Real=0.0, GWP₁₀₀::Real)
    m_lost = C_start + C_purchased - C_end - C_recovered - C_disposed
    (E=m_lost * GWP₁₀₀ / 1000, m_lost=m_lost)
end

"""
    fugitive_ldar(n_components; leak_rate, gwp)

Component-count method used when leaks are quantified by survey rather than by
mass balance: `m_CH₄ = N · leak_rate`, `E = GWP_CH₄ · m_CH₄`.
"""
function fugitive_ldar(n_components::Real; leak_rate::Real, gwp::GWPSet=GWP_AR6)
    m_CH₄ = n_components * leak_rate
    (E=co2e(; CH₄=m_CH₄, gwp=gwp) / 1000, m_CH₄=m_CH₄)
end

"""
    process_emissions(production; EF, gas, unit, gwp) -> NamedTuple

Process emissions from stoichiometry, `m = P · EF`, with the factor expressed per
tonne of product (`tCO₂/t` for clinker and lime, `kgN₂O/t` for nitric acid,
`tCO₂e/t` for ammonia and aluminium).
"""
function process_emissions(P::Real; EF::Real, gas::AbstractString="CO2", gwp::GWPSet=GWP_AR6,
                           unit::AbstractString="tCO₂/t")
    m = P * EF
    scale = parse_compound(unit).num.factor       # kg → t (1e-3), t → t (1.0)
    (gas == "CO2" || gas == "CO2e") && return (E=m * scale, m=m)
    (E=co2e(; CO₂=(gas == "CO2" ? m : 0.0), CH₄=(gas == "CH4" ? m : 0.0),
               N₂O=(gas == "N2O" ? m : 0.0), gwp=gwp) * scale, m=m)
end


"Oxidation factors OFᵢ by fuel (IPCC defaults; 1.00 for gaseous fuels)."
const DEFAULT_OXIDATION = Dict("natural_gas" => 1.00, "lpg" => 1.00, "fuel_oil" => 0.99,
    "diesel_stationary" => 0.99, "coal" => 0.98, "biomass" => 0.99, "blast_furnace_gas" => 0.99)

"Combustion efficiencies η_b by fuel, used when the activity is *useful* energy."
const DEFAULT_EFFICIENCY = Dict("natural_gas" => 0.90, "lpg" => 0.90, "fuel_oil" => 0.85,
    "diesel_stationary" => 0.85, "coal" => 0.80, "biomass" => 0.75, "blast_furnace_gas" => 0.95)

"Stoichiometric biogenic CO₂ of solid biomass, for the memo item (kgCO₂/GJ, IPCC)."
const BIOGENIC_CO₂_PER_GJ = 100.0

"Relative standard uncertainty of an activity, by data-quality class DQ."
const DQ_UNCERTAINTY = Dict(1 => 0.01, 2 => 0.03, 3 => 0.07, 4 => 0.15)

"Relative uncertainty of an activity (1σ); a soft-sensor estimate overrides the DQ default."
u_activity(rec::ActivityRecord) = get(rec.meta, "u_rel", "") == "" ?
    DQ_UNCERTAINTY[clamp(rec.DQ, 1, 4)] : parse(Float64, rec.meta["u_rel"])

"Which Scope-1 sub-total a source type belongs to."
const SCOPE1_BUCKET = Dict(
    "natural_gas" => "E_stat", "fuel_oil" => "E_stat", "diesel_stationary" => "E_stat",
    "lpg" => "E_stat", "coal" => "E_stat", "biomass" => "E_stat", "blast_furnace_gas" => "E_stat",
    "mobile_combustion" => "E_mob", "diesel_fleet" => "E_mob", "gasoline_fleet" => "E_mob",
    "aviation_fuel" => "E_mob", "marine_fuel" => "E_mob",
    "refrigerant" => "E_fug", "ch4_vent" => "E_fug", "ldar" => "E_fug",
    "compressor_seal" => "E_fug", "flaring" => "E_flare",
    "process_cement" => "E_proc", "process_lime" => "E_proc", "process_nitric_acid" => "E_proc",
    "process_ammonia" => "E_proc", "process_aluminium" => "E_proc")

"Fuels whose activity is an energy input and therefore get the gas-triple treatment."
const STATIONARY_FUELS = Set(["natural_gas", "fuel_oil", "diesel_stationary", "lpg", "coal",
    "biomass", "blast_furnace_gas"])

"Rules priced with a single CO₂e factor per litre, per kg or per m³."
const SINGLE_FACTOR_RULES = Set(["mobile_combustion", "diesel_fleet", "gasoline_fleet",
    "aviation_fuel", "marine_fuel", "ch4_vent", "waste", "end_of_life"])

"""
    pick_factor(library, rec, items; region, date, gas)

First factor that is both registered for a candidate category/item and
*dimensionally compatible* with the unit in which the activity was measured —
this is what lets an `MWh`-based and a `GJ`-based extract share one library.
Candidates are tried in the order given, with `rec.source_type` as the first
category and the mobile-combustion set as fallback.
"""
function pick_factor(lib::FactorLibrary, rec::ActivityRecord,
                     items::AbstractVector{<:AbstractString};
                     region::AbstractString="GLOBAL", date::Date=today(),
                     gas::Union{Nothing,AbstractString}=nothing)
    cats = unique(vcat([rec.source_type], ["mobile_combustion"]))
    for item in items, cat in cats
        for f in lookup_factor(lib, cat, item; region=region, date=date, gas=gas)
            compatibility(f, rec.Q̇ᵢ.code) && return f
        end
    end
    # last resort: any factor of these categories that is dimensionally compatible
    for cat in cats
        for f in lookup_factor_cat(lib, cat; region=region, date=date)
            (gas === nothing || f.gas == gas) || continue
            compatibility(f, rec.Q̇ᵢ.code) && return f
        end
    end
    nothing
end

"""
    scope1_row(rec, library; kwargs...) -> NamedTuple

Route one Scope-1 activity to the right formula and report which one was used:

| source type                | formula |
|----------------------------|---------|
| stationary fuels (energy)  | [`combustion_emissions`](@ref), gas by gas, with OF and η_b |
| fleets, vents, waste       | single `E = Q̇ᵢ · EF_CO₂e` |
| refrigerant, seal loss     | mass balance × GWP₁₀₀ |
| LDAR component count       | `N · leak_rate · GWP_CH₄` |
| flaring                    | [`flare_emissions`](@ref) with DRE and η_comb |
| process (clinker, acid, …) | [`process_emissions`](@ref), stoichiometric |
"""
function scope1_row(rec::ActivityRecord, lib::FactorLibrary;
                    region::AbstractString="GLOBAL", date::Date=today(), gwp::GWPSet=GWP_AR6,
                    of_table::AbstractDict=DEFAULT_OXIDATION,
                    eff_table::AbstractDict=DEFAULT_EFFICIENCY,
                    flare_params::AbstractDict=Dict{String,Float64}())
    st = rec.source_type
    rule = get(SOURCE_RULES, st, nothing)
    cat = rule === nothing ? st : rule.key
    hint = match_item_by_text(lib, cat, rec.item)
    items = filter(!isempty, unique(vcat([rec.item], something(hint, ""),
        [st, get(DEFAULT_ITEM_BY_RULE, cat, ""), "diesel_fleet"])))
    u_or(f) = f === nothing ? 0.0 : f.uncertainty

    if st in STATIONARY_FUELS
        f_co₂ = pick_factor(lib, rec, items; region=region, date=date, gas="CO2")
        f_co₂ === nothing && throw(ArgumentError(
            "no CO₂ factor for «$st» compatible with «$(rec.Q̇ᵢ.code)» in library $(lib.name)"))
        f_ch₄ = pick_factor(lib, rec, items; region=region, date=date, gas="CH4")
        f_n₂o = pick_factor(lib, rec, items; region=region, date=date, gas="N2O")
        r = combustion_emissions(rec.Q̇_canon;
            EF_CO₂=f_co₂.value, EF_CH₄=f_ch₄ === nothing ? 0.0 : f_ch₄.value,
            EF_N₂O=f_n₂o === nothing ? 0.0 : f_n₂o.value,
            OF=get(of_table, st, 1.0), η_b=get(eff_table, st, 1.0), gwp=gwp,
            fossil=(st != "biomass"))
        memo = st == "biomass" ? r.Q̇_in * BIOGENIC_CO₂_PER_GJ / 1000 : 0.0
        return (E=r.E, method="IPCC tier 1 (gas-specific, OF, η_b)", f=f_co₂,
                u_rel=row_u_rel(row_u_rel(u_or(f_co₂), u_or(f_ch₄)), u_or(f_n₂o)),
                memo_biogenic=memo, detail=r)

    elseif st in ("refrigerant", "compressor_seal")
        f = pick_factor(lib, rec, items; region=region, date=date)
        f === nothing && (f = pick_factor(lib, rec, ["R134a"]; region=region, date=date))
        f === nothing && throw(ArgumentError("no GWP factor for refrigerant «$(rec.item)»"))
        E = emission(rec.Q̇ᵢ, Quantity(f.value, f.unit)).val
        return (E=E, method="mass balance × GWP₁₀₀", f=f,
                u_rel=row_u_rel(u_activity(rec), f.uncertainty), memo_biogenic=0.0, detail=nothing)

    elseif st == "ldar"
        f = pick_factor(lib, rec, items; region=region, date=date)
        f === nothing && throw(ArgumentError("no LDAR leak-rate factor in library $(lib.name)"))
        E = emission_gas(rec.Q̇ᵢ, Quantity(f.value, f.unit); gwp=gwp).val
        return (E=E, method="component-count (LDAR) × GWP_CH₄", f=f,
                u_rel=row_u_rel(u_activity(rec), f.uncertainty), memo_biogenic=0.0, detail=nothing)

    elseif st == "flaring"
        f = pick_factor(lib, rec, items; region=region, date=date, gas="CO2e")
        f === nothing && throw(ArgumentError("no flare-gas factor in library $(lib.name)"))
        DRE = get(flare_params, "DRE", parse(Float64, get(rec.meta, "DRE", "0.98")))
        r = flare_emissions(rec.Q̇_canon; EF_flare=f.value, DRE=DRE,
                            η_comb=get(flare_params, "η_comb", 0.995),
                            x_CH₄=get(flare_params, "x_CH₄", 0.80), gwp=gwp)
        return (E=r.E, method="flaring with DRE and η_comb", f=f,
                u_rel=row_u_rel(u_activity(rec), f.uncertainty), memo_biogenic=0.0, detail=r)

    elseif startswith(st, "process_")
        f = pick_factor(lib, rec, items; region=region, date=date)
        f === nothing && throw(ArgumentError("no process factor for «$st» in library $(lib.name)"))
        r = process_emissions(rec.Q̇_canon; EF=f.value, gas=f.gas, unit=f.unit, gwp=gwp)
        return (E=r.E, method="process stoichiometry", f=f,
                u_rel=row_u_rel(u_activity(rec), f.uncertainty), memo_biogenic=0.0, detail=r)
    end

    # single-factor rules: fleets, vents, waste, end-of-life
    f = pick_factor(lib, rec, items; region=region, date=date)
    f === nothing && throw(ArgumentError(
        "no factor for «$st» compatible with «$(rec.Q̇ᵢ.code)» in library $(lib.name)"))
    E = f.gas == "CO2e" ? emission(rec.Q̇ᵢ, Quantity(f.value, f.unit)).val :
                          emission_gas(rec.Q̇ᵢ, Quantity(f.value, f.unit); gwp=gwp).val
    (E=E, method=f.gas == "CO2e" ? "average factor (CO₂e)" : "gas-specific factor", f=f,
     u_rel=row_u_rel(u_activity(rec), f.uncertainty), memo_biogenic=0.0, detail=nothing)
end

"""
    compute_scope1(records; library, region, date, consolidation, …) -> Scope1Result

Direct emissions of the whole portfolio, with the consolidation share ©ᵢ applied
and the sub-totals `E_stat`, `E_mob`, `E_fug`, `E_flare`, `E_proc` kept apart.
Every engine run is written to the audit trail when a ledger is supplied.
"""
function compute_scope1(records::AbstractVector{ActivityRecord},
                        lib::FactorLibrary=default_library();
                        region::AbstractString="GLOBAL", date::Date=today(),
                        consolidation::Symbol=:operational,
                        shares::AbstractDict=Dict{String,Float64}(),
                        gwp::GWPSet=GWP_AR6,
                        of_table::AbstractDict=DEFAULT_OXIDATION,
                        eff_table::AbstractDict=DEFAULT_EFFICIENCY,
                        flare_params::AbstractDict=Dict{String,Float64}(),
                        ledger::Union{Nothing,AuditLedger}=nothing,
                        actor::Union{Nothing,Actor}=nothing,
                        run_context::AbstractString="scope 1 calculation run")
    rows, by_source, memo = EmissionRow[], Dict{String,Float64}(), 0.0
    for rec in records
        rec.scope == :scope1 || continue
        © = consolidate(consolidation, rec; shares=shares)
        d = scope1_row(rec, lib; region=region, date=date, gwp=gwp, of_table=of_table,
                       eff_table=eff_table, flare_params=flare_params)
        E = d.E * ©
        push!(rows, emission_row(rec, d.f; E=E, method=d.method, ©=©,
            u_act=u_activity(rec), u_fac=d.u_rel))
        bucket = get(SCOPE1_BUCKET, rec.source_type, "E_stat")
        by_source[bucket] = get(by_source, bucket, 0.0) + E
        memo += d.memo_biogenic * ©
    end
    E_Scope1 = Σᵢ(r -> r.E, rows)
    if ledger !== nothing && actor !== nothing
        record!(ledger, actor, "calculate", "scope1"; why=run_context,
                after=@sprintf("%.3f tCO₂e from %d row(s)", E_Scope1, length(rows)))
    end
    Scope1Result(rows, by_source, E_Scope1, memo)
end

"Scope-1 detail as a DataFrame, with the sub-total each row belongs to."
function scope1_detail(r::Scope1Result)
    df = emissions_dataframe(r.rows)
    isempty(df) && return df
    df.bucket = [get(SCOPE1_BUCKET, st, "E_stat") for st in df.source_type]
    select(df, :rec_id, :site, :source_type, :bucket, :item, :method, :Q̇, :unit,
           :ef_id, :ef_value, :ef_unit, :share, :E, :DQ, :u_rel)
end

# ══════════════════════ Scope 2 — purchased energy ═══════════════════════════
"""
    compute_scope2(records; library, date, consolidation, …) -> Scope2Result

Purchased electricity, steam, heat and cooling, reported **twice** as the GHG
Protocol requires:

    E_loc = Σᵢ ( Cᵢ · EF_grid,y )                                 location based
    E_mkt = Σᵢ ( Cᵢ · EF_contract ) + C_unclaimed · EF_residual    market based

For every purchased-energy row the engine emits a `location-based` row and a
`market-based` row, so both figures can be decomposed the same way. The market
factor is taken from the contractual instrument declared in the record
(`meta["instrument"]`, e.g. `ppa_renewable`, `green_tariff`); otherwise the
residual mix is applied. λ_cov reports how much of the consumption is covered by
instruments at all.
"""
function compute_scope2(records::AbstractVector{ActivityRecord},
                        lib::FactorLibrary=default_library();
                        date::Date=today(),
                        consolidation::Symbol=:operational,
                        shares::AbstractDict=Dict{String,Float64}(),
                        ledger::Union{Nothing,AuditLedger}=nothing,
                        actor::Union{Nothing,Actor}=nothing,
                        run_context::AbstractString="scope 2 calculation run")
    rows = EmissionRow[]
    E_loc, E_mkt, C_elec, covered = 0.0, 0.0, 0.0, 0.0
    per_site = Dict{Tuple{String,String},Vector{Float64}}()   # (site, country) → [MWh, loc, mkt]

    for rec in records
        rec.scope == :scope2 || continue
        rule = SOURCE_RULES[rec.source_type]
        © = consolidate(consolidation, rec; shares=shares)
        region = isempty(rec.country) ? "GLOBAL" : rec.country

        f_loc = resolve_factor(lib, rule; region=region, date=date, market=:location)
        f_loc === nothing && (f_loc = resolve_factor(lib, rule; region="GLOBAL", date=date, market=:location))
        f_loc === nothing && throw(ArgumentError(
            "no location-based factor for «$(rec.source_type)» in region $region"))

        instrument = get(rec.meta, "instrument", "")
        contractual = rec.source_type == "ppa_electricity" || !isempty(instrument)
        f_mkt = if contractual
            i = isempty(instrument) ? "ppa_renewable" : instrument
            m = resolve_factor(lib, SOURCE_RULES["ppa_electricity"]; item=i, region=region, date=date)
            (m === nothing || !compatibility(m, rec.Q̇ᵢ.code)) ? f_loc : m
        else
            m = resolve_factor(lib, rule; region=region, date=date, market=:market)
            m === nothing ? f_loc : m
        end

        loc = emission(rec.Q̇ᵢ, Quantity(f_loc.value, f_loc.unit)).val
        mkt = emission(rec.Q̇ᵢ, Quantity(f_mkt.value, f_mkt.unit)).val

        push!(rows, emission_row(rec, f_loc; E=loc * ©, method="location-based", ©=©,
              u_act=u_activity(rec), u_fac=f_loc.uncertainty))
        push!(rows, emission_row(rec, f_mkt; E=mkt * ©, method="market-based", ©=©,
              u_act=u_activity(rec), u_fac=f_mkt.uncertainty))

        E_loc += loc * ©
        E_mkt += mkt * ©
        if rec.source_type in ("electricity", "ppa_electricity")
            C_elec += rec.Q̇_canon / 3.6              # GJ → MWh
            covered += contractual ? rec.Q̇_canon / 3.6 : 0.0
        end
        key = (rec.site_id, region)
        v = get!(per_site, key, Float64[0.0, 0.0, 0.0])
        v[1] += rec.Q̇_canon
        v[2] += loc * ©
        v[3] += mkt * ©
    end

    by_site = DataFrame(site=[k[1] for k in keys(per_site)], country=[k[2] for k in keys(per_site)],
        activity=[v[1] for v in values(per_site)], E_loc=[v[2] for v in values(per_site)],
        E_mkt=[v[3] for v in values(per_site)])
    isempty(by_site) || sort!(by_site, [:country, :site])
    λ_cov = C_elec > 0 ? covered / C_elec : 0.0

    if ledger !== nothing && actor !== nothing
        record!(ledger, actor, "calculate", "scope2"; why=run_context,
                after=@sprintf("location %.3f tCO₂e, market %.3f tCO₂e, λ_cov = %.1f%%",
                               E_loc, E_mkt, 100 * λ_cov))
    end
    Scope2Result(rows, E_loc, E_mkt, C_elec, λ_cov, by_site)
end

"Scope-2 detail as a DataFrame; `which ∈ {:location, :market, :both}`."
function scope2_detail(r::Scope2Result; which::Symbol=:both)
    df = emissions_dataframe(r.rows)
    isempty(df) && return df
    which == :both && return df
    filter(row -> row.method == (which == :location ? "location-based" : "market-based"), df)
end

# ══════════════════════ Scope 3 — value chain ════════════════════════════════
"Human-readable names of the 15 Scope-3 categories."
const SCOPE3_CATEGORY_NAMES = Dict(
    1 => "1 Purchased goods and services", 2 => "2 Capital goods",
    3 => "3 Fuel- and energy-related activities", 4 => "4 Upstream transportation and distribution",
    5 => "5 Waste generated in operations", 6 => "6 Business travel",
    7 => "7 Employee commuting", 8 => "8 Upstream leased assets",
    9 => "9 Downstream transportation and distribution", 10 => "10 Processing of sold products",
    11 => "11 Use of sold products", 12 => "12 End-of-life treatment of sold products",
    13 => "13 Downstream leased assets", 14 => "14 Franchises", 15 => "15 Investments")

"""
    scope3_method(rule, rec) -> String

Name the *method* behind a Scope-3 figure — the label an auditor reads first,
because data quality is what decides whether a hotspot may stay estimated.
"""
function scope3_method(rule::SourceRule, rec::ActivityRecord)
    if rule.dim == :currency
        return "spend based (EEIO, deflated)"
    elseif rec.DQ == 1
        return "supplier specific"
    elseif rec.DQ == 2
        return "average data (site measured)"
    elseif rec.DQ == 3
        return "average data (industry)"
    end
    "proxy estimate"
end

"""
    compute_scope3(records; library, date, consolidation, …) -> Scope3Result

Value-chain emissions for categories 1…15 from the same row machinery, with the
hybrid data hierarchy of the GHG Protocol:

    cat 1, 2, 8, 13, 14, 15   E = s_spend · EF_EEIO        spend based  (DQ 4, screening)
    cat 3                     E = Q̇_energy · EF_WTT        average data
    cat 4, 9                  E = tkm · EF                  distance based
    cat 5, 12                 E = m · EF_waste              waste-type specific
    cat 6, 7                  E = pkm · EF                  travel & commuting
    cat 10, 11                E = m or Q̇ · EF              processing / use phase

`screening` returns the Pareto hotspots: the categories that together make up
`hotspot_share` of the Scope-3 total and therefore must be refined with
supplier-specific data.
"""
function compute_scope3(records::AbstractVector{ActivityRecord},
                        lib::FactorLibrary=default_library();
                        date::Date=today(),
                        consolidation::Symbol=:operational,
                        shares::AbstractDict=Dict{String,Float64}(),
                        hotspot_share::Real=0.80,
                        ledger::Union{Nothing,AuditLedger}=nothing,
                        actor::Union{Nothing,Actor}=nothing,
                        run_context::AbstractString="scope 3 calculation run")
    rows = EmissionRow[]
    for rec in records
        rec.scope == :scope3 || continue
        rule = SOURCE_RULES[rec.source_type]
        © = consolidate(consolidation, rec; shares=shares)
        region = rule.key in EEIO_CATEGORIES ? "GLOBAL" : (isempty(rec.country) ? "GLOBAL" : rec.country)
        f = resolve_factor(lib, rule; item=isempty(rec.item) ? nothing : rec.item,
                           region=region, date=date)
        f === nothing && throw(ArgumentError(
            "no Scope-3 factor for «$(rec.source_type)» / item «$(rec.item)»"))
        E = emission(rec.Q̇ᵢ, Quantity(f.value, f.unit)).val * ©
        push!(rows, emission_row(rec, f; E=E, method=scope3_method(rule, rec), ©=©,
              u_act=u_activity(rec), u_fac=f.uncertainty))
    end

    E_Scope3 = Σᵢ(r -> r.E, rows)
    bycat = DataFrame(cat=Int[], category=String[], E=Float64[], share=Float64[],
                      rows=Int[], DQ=Float64[], u_rel=Float64[])
    for c in 1:15
        sub = [r for r in rows if r.cat == c]
        isempty(sub) && continue
        E = Σᵢ(r -> r.E, sub)
        push!(bycat, (c, SCOPE3_CATEGORY_NAMES[c], E, E_Scope3 > 0 ? E / E_Scope3 : 0.0,
                      length(sub), Σᵢ(r -> r.E * r.DQ, sub) / max(E, eps()), Σᵢ(r -> r.E * r.u_rel, sub) / max(E, eps())))
    end
    sort!(bycat, :cat)

    hotspots = String[]
    if !isempty(bycat)
        sorted = sort(bycat, :E, rev=true)
        cum = 0.0
        for r in eachrow(sorted)
            push!(hotspots, SCOPE3_CATEGORY_NAMES[r.cat])
            cum += r.share
            cum >= hotspot_share && break
        end
    end

    if ledger !== nothing && actor !== nothing
        record!(ledger, actor, "calculate", "scope3"; why=run_context,
                after=@sprintf("%.3f tCO₂e over %d categor(y/ies); hotspots: %s",
                               E_Scope3, size(bycat, 1), join(hotspots, "; ")))
    end
    Scope3Result(rows, bycat, E_Scope3, hotspots)
end

"Scope-3 detail as a DataFrame."
scope3_detail(r::Scope3Result) = emissions_dataframe(r.rows)

# ══════════════════════ the inventory ════════════════════════════════════════
"""
    compute_inventory(records; library, consolidation, date, ledger, actor, …) -> InventoryResult

Run the three engines over the same record set and consolidate:

    E_total = E_Scope1 + E_Scope2 (location based) + E_Scope3

The market-based Scope 2 figure is reported next to it, never silently replacing
the location-based one. Sub-totals are written to the audit trail as they are
produced, so the inventory's own construction is traceable.
"""
function compute_inventory(records::AbstractVector{ActivityRecord},
                           lib::FactorLibrary=default_library();
                           consolidation::Symbol=:operational,
                           shares::AbstractDict=Dict{String,Float64}(),
                           date::Date=today(), region::AbstractString="GLOBAL",
                           gwp::GWPSet=GWP_AR6,
                           of_table::AbstractDict=DEFAULT_OXIDATION,
                           eff_table::AbstractDict=DEFAULT_EFFICIENCY,
                           flare_params::AbstractDict=Dict{String,Float64}(),
                           hotspot_share::Real=0.80,
                           ledger::Union{Nothing,AuditLedger}=nothing,
                           actor::Union{Nothing,Actor}=nothing)
    s1 = compute_scope1(records, lib; region=region, date=date, consolidation=consolidation,
                        shares=shares, gwp=gwp, of_table=of_table, eff_table=eff_table,
                        flare_params=flare_params, ledger=ledger, actor=actor)
    s2 = compute_scope2(records, lib; date=date, consolidation=consolidation, shares=shares,
                        ledger=ledger, actor=actor)
    s3 = compute_scope3(records, lib; date=date, consolidation=consolidation, shares=shares,
                        hotspot_share=hotspot_share, ledger=ledger, actor=actor)
    total = s1.E_Scope1 + s2.E_loc + s3.E_Scope3
    InventoryResult(s1, s2, s3, consolidation, s1.E_Scope1, s2.E_loc, s2.E_mkt, s3.E_Scope3,
                    total, vcat(s1.rows, s2.rows, s3.rows))
end

"All rows of an inventory as one DataFrame (the audit file of the calculation run)."
inventory_rows(r::InventoryResult) = emissions_dataframe(r.rows)

"""
    total_rows(r::InventoryResult) -> Vector{EmissionRow}

The rows that **sum to `E_total`**: Scope 1, the *location-based* Scope 2 rows and
Scope 3. `r.rows` additionally holds the market-based Scope 2 rows (the dual report
the GHG Protocol requires), which must never be added to the location-based figure —
this accessor is the one to use for uncertainty, quality and export aggregates.
"""
total_rows(r::InventoryResult) = [row for row in r.rows if row.method != "market-based"]

"""
    contribution_table(r::InventoryResult) -> DataFrame

The one table every stakeholder asks for first: the share of each scope and
sub-total, sized in tCO₂e.
"""
function contribution_table(r::InventoryResult)
    parts = [("Scope 1", r.E_Scope1), ("Scope 2 (location based)", r.E_Scope2_loc),
             ("Scope 2 (market based)", r.E_Scope2_mkt), ("Scope 3", r.E_Scope3)]
    total = r.E_total
    DataFrame(component=[p[1] for p in parts], E=[p[2] for p in parts],
              share=[total > 0 ? p[2] / total : 0.0 for p in parts])
end

"Scope-1 sub-totals as a DataFrame."
function scope1_breakdown(r::InventoryResult)
    s = r.scope1
    names = Dict("E_stat" => "stationary combustion", "E_mob" => "mobile combustion",
                 "E_fug" => "fugitive", "E_flare" => "flaring", "E_proc" => "process")
    keys_ = sort(collect(keys(s.by_source)))
    DataFrame(component=[get(names, k, k) for k in keys_],
              E=[s.by_source[k] for k in keys_],
              share=[s.E_Scope1 > 0 ? s.by_source[k] / s.E_Scope1 : 0.0 for k in keys_])
end
