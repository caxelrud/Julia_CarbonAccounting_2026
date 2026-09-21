# ══════════════════════════════════════════════════════════════════════════════
#  factors.jl — emission-factor libraries
#
#  An emission factor is not a number, it is a *claim*: it holds for a given
#  activity, region, gas and time window, it was published by somebody, and it
#  carries an uncertainty. This module therefore stores every factor as an
#  immutable `EFRecord` with its provenance and validity window, and resolves a
#  factor for a given activity by (library preference, region, date).
#
#  The built-in library reproduces the default values of
#      IPCC 2006 Guidelines (Vol. 2 Energy, Vol. 3 IPPU)
#      DEFRA/BEIS GHG conversion factors
#      US EPA Emission Factor Hub (nitric acid, landfill, mobile)
#      EPA/EEIO supply-chain factors for spend-based Scope 3
#      IEA/Ember grid intensities and residual mixes
#  and is deliberately versioned: changing a factor creates a new version, it
#  does not overwrite history — which is what an auditor expects to see.
# ══════════════════════════════════════════════════════════════════════════════

"""
    EFRecord

One published emission factor. `gas` is `"CO2e"` for CO₂-equivalent factors and a
species name (`"CO2"`, `"CH4"`, `"N2O"`) for gas-specific ones; `unit` is the
compound unit (`"kgCO₂e/kWh"`, `"kgCH₄/t"`, …) understood by [`parse_compound`](@ref).
"""
struct EFRecord
    ef_id::String
    library::String
    version::String
    category::String          # SourceRule key the factor applies to
    item::String              # fuel, refrigerant, material, service
    gas::String
    value::Float64
    unit::String
    region::String            # "GLOBAL", "BRA", "DEU", …
    gwp::String               # which GWP set the CO₂e value relies on
    valid_from::Date
    valid_to::Date
    DQ::Int
    uncertainty::Float64      # relative standard uncertainty σ/μ (1σ)
    source::String            # publication of the value
    note::String
end

"Compact rendering of one factor."
function Base.show(io::IO, f::EFRecord)
    print(io, f.ef_id, "  ", f.item, " → ", f.gas, " ", f.value, " ", f.unit,
          "  [", f.region, ", ", f.library, " ", f.version, ", ", f.valid_from, "…", f.valid_to,
          ", u_r=", round(100 * f.uncertainty; digits=1), "%]")
end

"Whether a factor claim is valid on `date`."
applies_on(f::EFRecord, date::Date) = f.valid_from <= date <= f.valid_to

"""
    FactorLibrary(name, version, records)

A versioned set of [`EFRecord`](@ref)s. Lookups never modify it: a change of a
factor is a new record in a new version.
"""
struct FactorLibrary
    name::String
    version::String
    records::Vector{EFRecord}
end
Base.length(l::FactorLibrary) = length(l.records)

"""
    factors_dataframe(library) -> DataFrame

The library as a table — the form in which factor sets are reviewed and approved.
"""
factors_dataframe(l::FactorLibrary) = DataFrame(
    ef_id=[f.ef_id for f in l.records], library=[f.library for f in l.records],
    version=[f.version for f in l.records], category=[f.category for f in l.records],
    item=[f.item for f in l.records], gas=[f.gas for f in l.records],
    value=[f.value for f in l.records], unit=[f.unit for f in l.records],
    region=[f.region for f in l.records], gwp=[f.gwp for f in l.records],
    valid_from=[f.valid_from for f in l.records], valid_to=[f.valid_to for f in l.records],
    DQ=[f.DQ for f in l.records], uncertainty=[f.uncertainty for f in l.records],
    source=[f.source for f in l.records], note=[f.note for f in l.records])

"""
    load_factors(path; name) -> FactorLibrary

Load a factor set from CSV or JSON (columns as in [`factors_dataframe`](@ref)); the
complement of `export_factors`.
"""
function load_factors(path::AbstractString; name::Union{Nothing,AbstractString}=nothing)
    recs = if endswith(lowercase(path), ".json")
        d = JSON.parsefile(path)
        [EFRecord(string(r["ef_id"]), string(r["library"]), string(r["version"]),
                  string(r["category"]), string(r["item"]), string(r["gas"]), Float64(r["value"]),
                  string(r["unit"]), string(r["region"]), string(r["gwp"]),
                  Date(r["valid_from"]), Date(r["valid_to"]), Int(r["DQ"]),
                  Float64(r["uncertainty"]), string(r["source"]), string(get(r, "note", "")))
         for r in (d isa AbstractDict ? d["factors"] : d)]
    else
        df = CSV.File(path) |> DataFrame
        [EFRecord(string(r.ef_id), string(r.library), string(r.version), string(r.category),
                  string(r.item), string(r.gas), Float64(r.value), string(r.unit),
                  string(r.region), string(r.gwp), Date(r.valid_from), Date(r.valid_to),
                  Int(r.DQ), Float64(r.uncertainty), string(r.source), string(r.note))
         for r in eachrow(df)]
    end
    FactorLibrary(name === nothing ? basename(path) : String(name),
                  isempty(recs) ? "0" : recs[1].version, recs)
end

"Write a factor library to CSV or JSON."
function export_factors(l::FactorLibrary, path::AbstractString)
    mkpath(dirname(path))
    if endswith(lowercase(path), ".json")
        write(path, JSON.json([Dict("ef_id" => f.ef_id, "library" => f.library, "version" => f.version,
            "category" => f.category, "item" => f.item, "gas" => f.gas, "value" => f.value,
            "unit" => f.unit, "region" => f.region, "gwp" => f.gwp,
            "valid_from" => string(f.valid_from), "valid_to" => string(f.valid_to),
            "DQ" => f.DQ, "uncertainty" => f.uncertainty, "source" => f.source,
            "note" => f.note) for f in l.records], 2))
    else
        CSV.write(path, factors_dataframe(l))
    end
    path
end

# ── the built-in factor set ──────────────────────────────────────────────────
"""
    _default_factor_records() -> Vector{EFRecord}

The built-in factor library. Values are the widely used published defaults, so
the repository runs without any licensed dataset; in production the same records
are replaced (or extended) by the licensed set through [`load_factors`](@ref).

Provenance of the defaults
  * combustion — IPCC 2006 Guidelines, Vol. 2 (Energy), Tables 1.4 / 2.2, net CV
  * fleet & travel — DEFRA/BEIS GHG conversion factors, 2023 (CO₂e incl. CH₄, N₂O)
  * grid intensity & residual mix — IEA/Ember 2023 averages
  * process — IPCC 2006 Vol. 3 (IPPU), US EPA Emission Factor Hub
  * supply chain — US EPA EEIO factors (deflated to USD 2024)
"""
function _default_factor_records()
    R = EFRecord[]
    const_lib, const_ver = "CarbonAccounting built-in", "2026.1"
    from, to = Date(2024, 1, 1), Date(2030, 12, 31)
    function add(category, item, gas, value, unit_, region, source, DQ, ur; gwp="AR6",
                 note="", valid_from=from, valid_to=to, library=const_lib, version=const_ver)
        push!(R, EFRecord(string(category, "_", item, "_", gas, "_", region, "_", version),
            library, version, category, item, gas, Float64(value), unit_, region, gwp,
            valid_from, valid_to, DQ, Float64(ur), source, note))
    end

    # ── Scope 1: stationary combustion, gas-specific, per GJ (net CV) ────────
    combustion = [
        ("natural_gas",       56.1,  1.0e-3, 1.0e-4, 0.03, 0.50, 0.50, "IPCC 2006 Vol.2 Tab.2.2"),
        ("fuel_oil",          77.4,  3.0e-3, 6.0e-4, 0.03, 0.50, 0.50, "IPCC 2006 Vol.2 Tab.2.2"),
        ("diesel_stationary", 74.1,  3.0e-3, 6.0e-4, 0.02, 0.50, 0.50, "IPCC 2006 Vol.2 Tab.2.2"),
        ("lpg",               63.1,  1.0e-3, 1.0e-4, 0.03, 0.50, 0.50, "IPCC 2006 Vol.2 Tab.2.2"),
        ("coal",              94.6,  1.0e-2, 1.5e-3, 0.03, 0.30, 0.30, "IPCC 2006 Vol.2 Tab.2.2"),
        ("biomass",            0.0,  3.0e-2, 4.0e-3, 0.05, 0.50, 0.50, "IPCC 2006 (biogenic CO₂ = memo)"),
        ("blast_furnace_gas", 259.6, 1.0e-3, 1.0e-4, 0.05, 0.50, 0.50, "IPCC 2006 Vol.2 Tab.2.2"),
    ]
    for (item, co2, ch4, n2o, ur1, ur2, ur3, src) in combustion
        note = item == "biomass" ?
            "biogenic CO₂ is excluded from Scope 1 by convention and reported as a memo item" : ""
        add(item, item, "CO2", co2,
            "kgCO₂/GJ", "GLOBAL", src, 2, ur1; note=note)
        add(item, item, "CH4", ch4, "kgCH₄/GJ", "GLOBAL", src, 2, ur2)
        add(item, item, "N2O", n2o, "kgN₂O/GJ", "GLOBAL", src, 2, ur3)
    end
    # mobile combustion, volumetric (DEFRA CO₂e per litre / per kg)
    for (item, ef, unit_, ur, src) in [
            ("diesel_fleet", 2.6998, "kgCO₂e/L", 0.03, "DEFRA 2023 diesel (average biofuel blend)"),
            ("gasoline_fleet", 2.3143, "kgCO₂e/L", 0.03, "DEFRA 2023 petrol (average biofuel blend)"),
            ("fuel_oil", 3.1797, "kgCO₂e/L", 0.03, "DEFRA 2023 burning oil"),
            ("aviation_fuel", 3.1500, "kgCO₂e/kg", 0.04, "DEFRA 2023 aviation turbine fuel"),
            ("marine_fuel", 3.2060, "kgCO₂e/kg", 0.05, "DEFRA 2023 marine diesel/gas oil")]
        add("mobile_combustion", item, "CO2e", ef, unit_, "GLOBAL", src, 2, ur)
    end
    # ── Scope 2: purchased electricity, location-based ───────────────────────
    for (region, ef, ur) in [("BRA", 0.0870, 0.10), ("DEU", 0.3800, 0.05), ("ZAF", 0.9500, 0.06),
                             ("USA", 0.3690, 0.06), ("CHN", 0.5800, 0.08), ("NLD", 0.3280, 0.05),
                             ("ESP", 0.1900, 0.06), ("FRA", 0.0560, 0.05), ("GBR", 0.2070, 0.05),
                             ("POL", 0.6620, 0.05), ("GLOBAL", 0.4360, 0.10)]
        add("electricity", "grid_average", "CO2e", ef, "kgCO₂e/kWh", region,
            "IEA/Ember 2023 grid intensity", 3, ur;
            note="location-based Scope 2; use hourly data where available")
    end
    # ── Scope 2: residual mix (market-based, uncovered energy) ──────────────
    for (region, ef) in [("BRA", 0.1500), ("DEU", 0.5690), ("ZAF", 1.1000), ("USA", 0.5200),
                         ("CHN", 0.7900), ("NLD", 0.4900), ("ESP", 0.2900), ("GBR", 0.3200),
                         ("GLOBAL", 0.6200)]
        add("electricity", "residual_mix", "CO2e", ef, "kgCO₂e/kWh", region,
            "AIB European residual mix / national disclosure 2023", 3, 0.15;
            note="applied to electricity not covered by contractual instruments")
    end
    # ── Scope 2: contractual instruments (market-based) ─────────────────────
    for (item, ef, ur, src, note) in [
            ("ppa_renewable", 0.0000, 0.00, "GHG Protocol Scope 2 Guidance (hierarchy)",
             "zero-emission contractual instrument; lifecycle emissions optional memo"),
            ("green_tariff", 0.0500, 0.50, "GHG Protocol Scope 2 Guidance",
             "tariff with certificates; factor from the supplier-specific disclosure"),
            ("unclaimed", 0.6200, 0.15, "GHG Protocol Scope 2 Guidance",
             "no instrument: residual-mix like treatment")]
        add("ppa_electricity", item, "CO2e", ef, "kgCO₂e/kWh", "GLOBAL", src, 2, ur; note=note)
    end
    # ── Scope 2: purchased steam, heat, cooling ─────────────────────────────
    for (cat, item, ef, unit_, ur, src) in [
            ("steam", "purchased_steam", 66.0, "kgCO₂e/GJ", 0.10, "gas boiler, η_b = 0.85 (56.1/0.85)"),
            ("heat", "district_heat", 70.0, "kgCO₂e/GJ", 0.20, "district heating mix, GLOBAL default"),
            ("cooling", "purchased_cooling", 30.0, "kgCO₂e/GJ", 0.25, "electric chiller, COP 3.5, grid mix")]
        add(cat, item, "CO2e", ef, unit_, "GLOBAL", src, 3, ur)
    end
    # ── Scope 1: fugitive emissions ─────────────────────────────────────────
    for (item, ef, ur, src, note) in [
            ("R134a", 1430.0, 0.10, "IPCC AR5 GWP₁₀₀", "charge-mass balance of the refrigerant bank"),
            ("R410A", 2088.0, 0.10, "IPCC AR5 GWP₁₀₀", "charge-mass balance of the refrigerant bank"),
            ("R404A", 3922.0, 0.10, "IPCC AR5 GWP₁₀₀", "high-GWP blend, phasing down"),
            ("R22", 1810.0, 0.10, "IPCC AR5 GWP₁₀₀", "HCFC, obsolete equipment"),
            ("R32", 675.0, 0.10, "IPCC AR5 GWP₁₀₀", "low-GWP blend component"),
            ("R744", 1.0, 0.05, "IPCC AR5 GWP₁₀₀", "CO₂ used as refrigerant"),
            ("NH3", 0.0, 0.05, "IPCC AR5 GWP₁₀₀", "ammonia, no direct GWP")]
        add("refrigerant", item, "CO2e", ef, "kgCO₂e/kg", "GLOBAL", src, 2, ur; note=note)
    end
    add("ch4_vent", "ch4_vented", "CO2e", 21.36, "kgCO₂e/m³(n)", "GLOBAL",
        "ρ_CH₄ · GWP₁₀₀ = 0.7168 kg/m³ × 29.8", 3, 0.40;
        note="top-down screening factor; a mass balance over the gas system is preferred")
    add("ldar", "component_leak", "CH4", 0.5, "kgCH₄/component", "GLOBAL",
        "EPA 40 CFR 98 Subpart W component-level average", 4, 0.60;
        note="per component-year; multiply by leak-survey results (see the anomaly notebook)")
    add("compressor_seal", "seal_loss", "CO2e", 25.0, "kgCO₂e/kg", "GLOBAL",
        "industry-average seal-loss factor", 4, 0.60)
    add("flaring", "flare_gas", "CO2e", 1.85, "kgCO₂e/m³(n)", "GLOBAL",
        "typical flare-gas composition, CC ≈ 0.65 tC/t", 2, 0.25;
        note="before destruction efficiency: the engine multiplies by DRE · η_comb")
    # ── Scope 1: process emissions (per tonne of product) ───────────────────
    for (item, gas, ef, unit_, ur, src) in [
            ("clinker", "CO2", 0.525, "tCO₂/t", 0.02, "IPCC 2006 Vol.3 Tab.2.1 (0.51–0.53)"),
            ("lime", "CO2", 0.785, "tCO₂/t", 0.03, "IPCC 2006 Vol.3 Tab.2.2"),
            ("nitric_acid", "N2O", 5.7, "kgN₂O/t", 0.40, "US EPA EF Hub, nitric acid (uncontrolled)"),
            ("ammonia", "CO2e", 1.90, "tCO₂e/t", 0.15, "IPCC 2006 Vol.3 (feedstock + fuel)"),
            ("aluminium", "CO2e", 1.60, "tCO₂e/t", 0.20, "IPCC 2006 Vol.3, anode + PFC (CF₄/C₂F₆)")]
        add(string("process_", item), item, gas, ef, unit_, "GLOBAL", src, 2, ur)
    end
    # ── Scope 3: upstream fuel and energy activities (WTT, cat 3) ───────────
    for (item, ef, unit_, ur) in [("natural_gas", 8.55, "kgCO₂e/GJ", 0.20),
                                  ("diesel_fleet", 16.20, "kgCO₂e/GJ", 0.20),
                                  ("gasoline_fleet", 15.60, "kgCO₂e/GJ", 0.20),
                                  ("coal", 5.50, "kgCO₂e/GJ", 0.25),
                                  ("aviation_fuel", 13.50, "kgCO₂e/GJ", 0.25),
                                  ("electricity_wtt", 0.045, "kgCO₂e/kWh", 0.30)]
        add("fuel_energy_upstream", item, "CO2e", ef, unit_, "GLOBAL",
            "DEFRA 2023 well-to-tank factors", 3, ur)
    end
    # ── Scope 3: transport (cat 4 and cat 9), per t·km ──────────────────────
    for cat in ("upstream_transport", "downstream_transport")
        for (item, ef, ur, src) in [("hgv_diesel", 0.107, 0.15, "DEFRA 2023 HGV (average laden)"),
                                    ("rail_freight", 0.028, 0.20, "DEFRA 2023 rail freight"),
                                    ("sea_container", 0.016, 0.25, "DEFRA 2023 container ship"),
                                    ("air_freight", 0.602, 0.20, "DEFRA 2023 air freight (long haul)"),
                                    ("inland_barge", 0.033, 0.25, "DEFRA 2023 inland waterway")]
            add(cat, item, "CO2e", ef, "kgCO₂e/t·km", "GLOBAL", src, 3, ur)
        end
    end
    # ── Scope 3: business travel (cat 6) and commuting (cat 7), per p·km ─────
    for cat in ("business_travel_air", "business_travel_land", "employee_commuting")
        for (item, ef, ur, src) in [("air_short_haul", 0.152, 0.20, "DEFRA 2023 domestic & short-haul air"),
                                    ("air_long_haul", 0.111, 0.20, "DEFRA 2023 long-haul air"),
                                    ("car_average", 0.171, 0.15, "DEFRA 2023 average car"),
                                    ("rail", 0.035, 0.25, "DEFRA 2023 national rail"),
                                    ("bus", 0.104, 0.25, "DEFRA 2023 local bus"),
                                    ("bicycle", 0.0, 0.05, "DEFRA 2023 active travel")]
            add(cat, item, "CO2e", ef, "kgCO₂e/p·km", "GLOBAL", src, 3, ur)
        end
    end
    # ── Scope 3: waste (cat 5) and end-of-life (cat 12), per kg ─────────────
    for cat in ("waste", "end_of_life")
        for (item, ef, ur, src) in [("landfill_msw", 0.467, 0.35, "US EPA WARM / DEFRA 2023 landfill"),
                                    ("recycling", 0.021, 0.40, "DEFRA 2023 open-loop recycling"),
                                    ("incineration", 0.021, 0.40, "DEFRA 2023 combustion of waste"),
                                    ("composting", 0.008, 0.50, "DEFRA 2023 composting"),
                                    ("wastewater", 0.270, 0.50, "IPCC 2006 Vol.5 wastewater")]
            add(cat, item, "CO2e", ef, "kgCO₂e/kg", "GLOBAL", src, 3, ur)
        end
    end
    # ── Scope 3: processing of sold products (cat 10), use phase (cat 11) ───
    add("processing_sold", "average_intermediate", "CO2e", 0.35, "kgCO₂e/kg", "GLOBAL",
        "average intermediate-processing factor", 4, 0.50)
    add("use_phase", "average_energy_use", "CO2e", 62.0, "kgCO₂e/GJ", "GLOBAL",
        "mixed energy carriers in the use phase", 3, 0.40)
    # ── Scope 3: spend-based EEIO factors (cats 1,2,8,13,14,15) ─────────────
    eeio = [("steel", 0.850, "EPA USEEIO, iron & steel"), ("cement", 0.650, "EPA USEEIO, cement"),
            ("chemicals", 0.420, "EPA USEEIO, basic chemicals"), ("plastics", 0.550, "EPA USEEIO, plastics"),
            ("electronics", 0.120, "EPA USEEIO, electronic components"),
            ("services", 0.035, "EPA USEEIO, professional services"),
            ("machinery", 0.180, "EPA USEEIO, machinery"), ("paper", 0.300, "EPA USEEIO, paper"),
            ("food", 0.750, "EPA USEEIO, food products"), ("textiles", 0.350, "EPA USEEIO, textiles"),
            ("mining", 0.400, "EPA USEEIO, mining"), ("transport_services", 0.350, "EPA USEEIO, transport"),
            ("electricity_supply", 0.800, "EPA USEEIO, electricity"),
            ("construction", 0.300, "EPA USEEIO, construction"),
            ("motor_vehicles", 0.220, "EPA USEEIO, motor vehicles"),
            ("pharmaceuticals", 0.150, "EPA USEEIO, pharmaceuticals"),
            ("agriculture", 0.500, "EPA USEEIO, agriculture")]
    for cat in ("purchased_goods_spend", "capital_goods", "upstream_leased",
                "downstream_leased", "franchises", "investments")
        for (item, ef, src) in eeio
            add(cat, item, "CO2e", ef, "kgCO₂e/USD2024", "GLOBAL", src, 4, 0.45;
                note="spend-based screening; refine hotspots with supplier-specific data")
        end
    end
    R
end

# ── library construction and factor resolution ───────────────────────────────
const _DEFAULT_LIBRARY = Ref{Union{Nothing,FactorLibrary}}(nothing)

"""
    default_library() -> FactorLibrary

The built-in library (`2026.1`), constructed once per session from
[`_default_factor_records`](@ref).
"""
function default_library()
    if _DEFAULT_LIBRARY[] === nothing
        _DEFAULT_LIBRARY[] = FactorLibrary("CarbonAccounting built-in", "2026.1",
                                           _default_factor_records())
    end
    _DEFAULT_LIBRARY[]
end

"""
    lookup_factor(library, category, item; gas, region, date) -> Vector{EFRecord}

All records of `category`/`item` valid on `date`, **ordered by precedence**:
the requested region first, then `GLOBAL`, then any other region; newest version
first inside each group.
"""
function lookup_factor(lib::FactorLibrary, category::AbstractString, item::AbstractString;
                       gas::Union{Nothing,AbstractString}=nothing,
                       region::AbstractString="GLOBAL", date::Date=today())
    c = [f for f in lib.records
         if f.category == category && f.item == item && applies_on(f, date) &&
            (gas === nothing || f.gas == gas)]
    rank(f) = (f.region == region ? 0 : (f.region == "GLOBAL" ? 1 : 2), f.version, f.library)
    sort!(c, by=rank)
    c
end

"All valid records of one category, in the same precedence order."
lookup_factor_cat(lib::FactorLibrary, category::AbstractString;
                  region::AbstractString="GLOBAL", date::Date=today()) =
    sort!([f for f in lib.records if f.category == category && applies_on(f, date)],
          by=f -> (f.region == region ? 0 : (f.region == "GLOBAL" ? 1 : 2), f.version))

"""
    match_item_by_text(library, category, text) -> Union{String,Nothing}

Find the item of a category whose name appears inside a free-text field, so that
`"R410A Nachfüllung"` resolves to the library item `R410A` and not to whichever
refrigerant happens to come first. The text is folded with [`header_key`](@ref).
"""
function match_item_by_text(lib::FactorLibrary, category::AbstractString, text::AbstractString)
    k = header_key(String(text))
    isempty(k) && return nothing
    for f in lookup_factor_cat(lib, category)
        item_key = header_key(f.item)
        (isempty(item_key) || item_key == k) && continue
        occursin(item_key, k) && return f.item
    end
    nothing
end

"""
Default item chosen for a rule when the activity record names no specific one:
a "waste" row without treatment route is priced with the landfill factor, a travel
row without cabin class with short-haul air, and so on.
"""
const DEFAULT_ITEM_BY_RULE = Dict{String,String}(
    "electricity" => "grid_average", "ppa_electricity" => "ppa_renewable",
    "steam" => "purchased_steam", "heat" => "district_heat", "cooling" => "purchased_cooling",
    "waste" => "landfill_msw", "end_of_life" => "landfill_msw",
    "upstream_transport" => "hgv_diesel", "downstream_transport" => "hgv_diesel",
    "business_travel_air" => "air_short_haul", "business_travel_land" => "car_average",
    "employee_commuting" => "car_average", "use_phase" => "average_energy_use",
    "processing_sold" => "average_intermediate", "refrigerant" => "R134a",
    "ch4_vent" => "ch4_vented", "ldar" => "component_leak", "compressor_seal" => "seal_loss",
    "flaring" => "flare_gas", "mobile_combustion" => "diesel_fleet",
    "purchased_goods_spend" => "services", "capital_goods" => "machinery",
    "upstream_leased" => "services", "downstream_leased" => "services",
    "franchises" => "services", "investments" => "services",
)

"Categories that carry a spend-based EEIO factor set instead of physical factors."
const EEIO_CATEGORIES = Set(["purchased_goods_spend", "capital_goods", "upstream_leased",
    "downstream_leased", "franchises", "investments"])

"""
    sector_hint(text) -> String

Map a free-text procurement description to the EEIO sector of the library
("stainless steel plate" → `steel`, "facility management" → `services`): the join
between an ERP purchase-order text and the emission-factor library.
"""
function sector_hint(text::AbstractString)
    k = header_key(text)
    for (needle, sector) in SECTOR_ALIASES
        occursin(needle, k) && return sector
    end
    "services"
end

const SECTOR_ALIASES = [
    ("steel", "steel"), ("aco", "steel"), ("iron", "steel"), ("cement", "cement"),
    ("clinker", "cement"), ("concrete", "cement"), ("chemical", "chemicals"),
    ("quimic", "chemicals"), ("solvent", "chemicals"), ("plastic", "plastics"),
    ("polimer", "plastics"), ("resin", "plastics"), ("electron", "electronics"),
    ("eletron", "electronics"), ("semiconductor", "electronics"), ("software", "services"),
    ("consult", "services"), ("consultoria", "services"), ("legal", "services"),
    ("facility", "services"), ("marketing", "services"), ("machin", "machinery"),
    ("equipment", "machinery"), ("pump", "machinery"), ("paper", "paper"),
    ("pulp", "paper"), ("food", "food"), ("alimento", "food"), ("textile", "textiles"),
    ("cotton", "textiles"), ("mining", "mining"), ("ore", "mining"),
    ("logistic", "transport_services"), ("freight", "transport_services"),
    ("transport_service", "transport_services"), ("electricity_supply", "electricity_supply"),
    ("construction", "construction"), ("building_works", "construction"),
    ("vehicle", "motor_vehicles"), ("fleet_purchase", "motor_vehicles"),
    ("pharma", "pharmaceuticals"), ("agricultur", "agriculture"), ("fertilizer", "agriculture"),
]

"""
    resolve_factor(library, rule::SourceRule; item, region, date, market) -> EFRecord or nothing

Pick the factor for one activity of a given [`SourceRule`](@ref):

  * electricity — location-based (`grid_average`) or market-based (`residual_mix`),
    selected by `market ∈ {:location, :market}`;
  * physical rules — exact `item` match, else the rule's default item, with the
    region/version precedence of [`lookup_factor`](@ref);
  * spend-based rules — the EEIO factor of the sector hinted by the item text;
  * last resort — the only valid record of the rule's category.
"""
function resolve_factor(lib::FactorLibrary, rule::SourceRule;
                        item::Union{Nothing,AbstractString}=nothing,
                        region::AbstractString="GLOBAL", date::Date=today(),
                        market::Symbol=:location)
    if rule.key == "electricity"
        c = lookup_factor(lib, "electricity",
                          market == :market ? "residual_mix" : "grid_average";
                          region=region, date=date)
        return isempty(c) ? nothing : first(c)
    end
    want = item === nothing ? get(DEFAULT_ITEM_BY_RULE, rule.key, nothing) : String(item)
    if want !== nothing
        c = lookup_factor(lib, rule.key, want; region=region, date=date)
        isempty(c) || return first(c)
    end
    if rule.key in EEIO_CATEGORIES
        c = lookup_factor(lib, rule.key, sector_hint(something(want, rule.key)); region=region, date=date)
        isempty(c) || return first(c)
    end
    c = lookup_factor(lib, "mobile_combustion", something(want, "diesel_fleet");
                      region=region, date=date)
    isempty(c) || return first(c)
    anyc = lookup_factor_cat(lib, rule.key; region=region, date=date)
    isempty(anyc) ? nothing : first(anyc)
end

"""
    compatibility(f::EFRecord, activity_code) -> Bool

Dimensional check of a factor against the unit in which the activity is measured —
the guard that prevents `MWh × kgCO₂e/GJ`-style mistakes.
"""
function compatibility(f::EFRecord, activity_code::AbstractString)
    cu = try
        parse_compound(f.unit)
    catch
        return false
    end
    cu.den === nothing && return false
    try
        return unit(activity_code).dim == cu.den.dim
    catch
        return false
    end
end

"Relative standard uncertainty of a factor (σ/μ), for the uncertainty engine."
factor_uncertainty(f::EFRecord) = f.uncertainty

"Human-readable description of a factor, for reports and the audit trail."
function describe_factor(f::EFRecord)
    @sprintf("%s = %g %s [%s %s, %s, DQ%d, u_r=%.0f%%, %s]",
             f.item, f.value, f.unit, f.library, f.version, f.region, f.DQ,
             100 * f.uncertainty, f.source)
end
