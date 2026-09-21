# ══════════════════════════════════════════════════════════════════════════════
#  units.jl — dimensional bookkeeping  (part 1/2)
#
#  Carbon accounting fails on units far more often than on physics. Every
#  activity datum and every emission factor carries an explicit dimension and
#  the multiplication  E = Q̇ · EF  is checked for dimensional consistency.
#
#  Canonical unit per dimension:
#     mass_co2e  tCO₂e    mass     t        energy  GJ
#     volume     m³(n)    length   km       time    h
#     count      unit     currency USD2024  temp    K
# ══════════════════════════════════════════════════════════════════════════════

"""
    UnitDef(code, dim, factor, offset, latex)

One measurement unit. Conversion into the canonical unit of the same dimension
is affine:   x_canonical = factor · x + offset
(the offset is non-zero only for temperature scales).
"""
struct UnitDef
    code::String
    dim::Symbol
    factor::Float64
    offset::Float64
    latex::String
end
UnitDef(code, dim, factor; offset=0.0, latex="") = UnitDef(code, dim, factor, offset, latex)

"Canonical unit per dimension — the unit in which all engines compute."
const CANONICAL = Dict{Symbol,String}(
    :mass_co2e      => "tCO₂e",
    :mass           => "t",
    :energy         => "GJ",
    :volume         => "m³",
    :length         => "km",
    :transport_work => "t·km",
    :time           => "h",
    :count          => "unit",
    :currency       => "USD2024",
    :temperature    => "K",
)

const UNITS = Dict{String,UnitDef}()

"Register a unit code in the global registry (called once at package load)."
function register_unit!(code::AbstractString, dim::Symbol, factor::Real;
                        offset::Real=0.0, latex::AbstractString="")
    UNITS[String(code)] = UnitDef(String(code), dim, Float64(factor), Float64(offset), String(latex))
    String(code)
end

function _register_units!()
    empty!(UNITS)
    # ── mass expressed as CO₂-equivalent ─────────────────────────────────────
    register_unit!("gCO₂e", :mass_co2e, 1e-6, latex="gCO_2e")
    register_unit!("kgCO₂e", :mass_co2e, 1e-3, latex="kgCO_2e")
    register_unit!("tCO₂e", :mass_co2e, 1.0, latex="tCO_2e")
    register_unit!("ktCO₂e", :mass_co2e, 1e3)
    register_unit!("MtCO₂e", :mass_co2e, 1e6)
    # ── physical mass (gas, fuel, product) ───────────────────────────────────
    register_unit!("g", :mass, 1e-6)
    register_unit!("kg", :mass, 1e-3)
    register_unit!("t", :mass, 1.0)
    register_unit!("lb", :mass, 4.5359237e-4)
    register_unit!("ton_us", :mass, 0.90718474)      # short ton
    register_unit!("ton_uk", :mass, 1.0160469088)    # long ton
    # ── energy ───────────────────────────────────────────────────────────────
    register_unit!("J", :energy, 1e-9)
    register_unit!("kJ", :energy, 1e-6)
    register_unit!("MJ", :energy, 1e-3)
    register_unit!("GJ", :energy, 1.0)
    register_unit!("TJ", :energy, 1e3)
    register_unit!("kWh", :energy, 3.6e-3)
    register_unit!("MWh", :energy, 3.6)
    register_unit!("GWh", :energy, 3.6e3)
    register_unit!("TWh", :energy, 3.6e6)
    register_unit!("Btu", :energy, 1.0550559e-6)
    register_unit!("MMBtu", :energy, 1.0550559e-3)
    register_unit!("therm", :energy, 1.0550559e-1)
    register_unit!("toe", :energy, 41.868)           # tonne of oil equivalent
    register_unit!("koe", :energy, 4.1868e-2)
    # ── volume at normal conditions ──────────────────────────────────────────
    register_unit!("m³", :volume, 1.0, latex="m^3(n)")
    register_unit!("m³(n)", :volume, 1.0)
    register_unit!("L", :volume, 1e-3)
    register_unit!("scf", :volume, 2.8316847e-2)     # standard cubic foot
    register_unit!("MMscf", :volume, 2.8316847e4)
    register_unit!("bbl", :volume, 1.589873e-1)      # barrel of oil
    register_unit!("gal_us", :volume, 3.7854118e-3)
    register_unit!("ft³", :volume, 2.8316847e-2)
    # ── distance, time, count ────────────────────────────────────────────────
    register_unit!("m", :length, 1e-3)
    register_unit!("km", :length, 1.0)
    register_unit!("mile", :length, 1.609344)
    register_unit!("nmi", :length, 1.852)
    # ── transport work (the activity of Scope-3 transport categories) ────────
    register_unit!("t·km", :transport_work, 1.0, latex="t\\!\\cdot\\!km")
    register_unit!("p·km", :transport_work, 1.0, latex="p\\!\\cdot\\!km")
    register_unit!("ton_mile", :transport_work, 1.609344)
    register_unit!("h", :time, 1.0)
    register_unit!("day", :time, 24.0)
    register_unit!("week", :time, 168.0)
    register_unit!("year", :time, 8760.0)
    register_unit!("unit", :count, 1.0)
    register_unit!("head", :count, 1.0)              # livestock
    register_unit!("component", :count, 1.0)         # LDAR component count
    # ── money (deflated to the EEIO base year by deflect_to_base) ────────────
    register_unit!("USD2024", :currency, 1.0, latex="USD_{2024}")
    register_unit!("USD", :currency, 1.0)
    register_unit!("EUR", :currency, 1.10)
    register_unit!("CNY", :currency, 0.14)
    register_unit!("BRL", :currency, 0.20)
    register_unit!("ZAR", :currency, 0.055)
    # ── temperature (affine scale!) ──────────────────────────────────────────
    register_unit!("K", :temperature, 1.0, offset=0.0)
    register_unit!("°C", :temperature, 1.0, offset=273.15, latex="{}^{\\circ}C")
    register_unit!("°F", :temperature, 5 / 9, offset=255.3722222, latex="{}^{\\circ}F")
    return UNITS
end

_register_units!()

const UNIT_ALIASES = Dict{String,String}(
    "m3" => "m³", "m3(n)" => "m³(n)", "kwh" => "kWh", "mwh" => "MWh",
    "gCO2e" => "gCO₂e", "kgCO2e" => "kgCO₂e", "tCO2e" => "tCO₂e",
    "ktCO2e" => "ktCO₂e", "MtCO2e" => "MtCO₂e", "tco2e" => "tCO₂e",
    "celcius" => "°C", "celsius" => "°C", "C" => "°C", "fahrenheit" => "°F",
    "short_ton" => "ton_us", "long_ton" => "ton_uk", "tonne" => "t", "MT" => "t",
    "metric_ton" => "t", "usd" => "USD", "eur" => "EUR", "cny" => "CNY",
    "brl" => "BRL", "zar" => "ZAR", "unitless" => "unit", "num" => "unit",
    "pkm" => "p·km", "tkm" => "t·km", "tonmile" => "ton_mile", "kWh_th" => "kWh", "MWh_th" => "MWh",
    "std_m3" => "m³", "scm" => "m³", "Nm3" => "m³(n)", "nm3" => "m³(n)",
)

"""
    unit(code) -> UnitDef

Unit lookup that accepts the spellings found in real ERP/SCADA exports
(`m3`, `Nm3`, `tCO2e`, `tonne`, `kwh`, …).
"""
function unit(code::AbstractString)::UnitDef
    c = strip(String(code))
    haskey(UNITS, c) && return UNITS[c]
    haskey(UNIT_ALIASES, c) && return UNITS[UNIT_ALIASES[c]]
    lc = lowercase(c)
    for (k, v) in UNITS
        lowercase(k) == lc && return v
    end
    throw(ArgumentError("unknown unit «$code» — register it in _register_units! (src/units.jl)"))
end

"Whether `code` names a known unit."
has_unit(code::AbstractString) = try
    unit(code); true
catch
    false
end

"""
    Quantity(val, code)          e.g. Quantity(12.5, "MWh")
    q"12.5 MWh"  /  u"12.5 MWh"  (string macros, read like the raw ERP export)

A number *with* its unit. All arithmetic checks the dimension, so a formula can
be read exactly as written on paper:

    Eᵢ = Q̇ᵢ · EFᵢ        ⟶   Eᵢ = emission(q"1250 MWh", q"0.42 kgCO₂e/kWh")
"""
struct Quantity{T<:Real}
    val::T
    code::String
end
Quantity(val::Real, code::AbstractString) = Quantity{typeof(float(val))}(float(val), String(code))

"Parse «12.5 MWh», «1,250 kWh», «-3.2 °C», «75 tCO2e» or «0.42 kgCO2e/kWh» into a `Quantity`."
function parse_quantity(str::AbstractString)
    m = match(r"^\s*([-+]?[0-9][0-9_,.]*(?:[eE][-+]?[0-9]+)?)\s*(.*?)\s*$", String(str))
    m === nothing && throw(ArgumentError("cannot parse quantity «$str»"))
    v = parse_number(m.captures[1])          # tolerates 1,234.5 and 1.234,5
    raw = m.captures[2]
    code = if isempty(raw)
        "unit"
    elseif has_unit(raw)
        unit(raw).code
    else
        # not a plain unit — accept a compound factor unit such as «kgCO₂e/kWh»
        parse_compound(raw).raw
    end
    Quantity(v, code)
end

macro q_str(s::String)
    return :(parse_quantity($s))
end

"`u\"12.5 MWh\"` — alias of `q\"…\"`, echoing the spelling of the source export."
macro u_str(s::String)
    return :(parse_quantity($s))
end

unitdef(q::Quantity) = unit(q.code)
dimension(q::Quantity) = unitdef(q).dim

"Convert a quantity to another unit of the same dimension (affine for temperature)."
function value_in(q::Quantity, code::AbstractString)::Float64
    a, b = unitdef(q), unit(code)
    a.dim == b.dim || throw(DimensionMismatch("«$(q.code)» is $(a.dim) but «$code» is $(b.dim)"))
    (q.val * a.factor + a.offset - b.offset) / b.factor
end

"Value of `q` in the canonical unit of its dimension (t, GJ, m³, tCO₂e, …)."
val_canon(q::Quantity) = value_in(q, CANONICAL[dimension(q)])

"Is `q` already expressed in the canonical unit of its dimension?"
is_canonical(q::Quantity) = q.code == CANONICAL[dimension(q)]

"Same value, expressed in the canonical unit of its dimension."
canonical(q::Quantity) = Quantity(val_canon(q), CANONICAL[dimension(q)])

function _same_dim(a::Quantity, b::Quantity)
    dimension(a) == dimension(b) || throw(DimensionMismatch(
        "cannot combine $(a.code) ($(dimension(a))) with $(b.code) ($(dimension(b)))"))
    nothing
end

Base.:+(a::Quantity, b::Quantity) = (_same_dim(a, b); Quantity(a.val + value_in(b, a.code), a.code))
Base.:-(a::Quantity, b::Quantity) = (_same_dim(a, b); Quantity(a.val - value_in(b, a.code), a.code))
Base.:-(a::Quantity) = Quantity(-a.val, a.code)
Base.:*(a::Quantity, s::Real) = Quantity(a.val * s, a.code)
Base.:*(s::Real, a::Quantity) = Quantity(a.val * s, a.code)
Base.:/(a::Quantity, s::Real) = Quantity(a.val / s, a.code)
Base.:/(a::Quantity, b::Quantity) = (_same_dim(a, b); a.val / value_in(b, a.code))
Base.:(==)(a::Quantity, b::Quantity) = dimension(a) == dimension(b) && (a.val == value_in(b, a.code))
Base.isless(a::Quantity, b::Quantity) = (_same_dim(a, b); isless(a.val, value_in(b, a.code)))
Base.isapprox(a::Quantity, b::Quantity; kwargs...) = (_same_dim(a, b); isapprox(a.val, value_in(b, a.code); kwargs...))
Base.abs(a::Quantity) = Quantity(abs(a.val), a.code)
Base.zero(a::Quantity) = Quantity(0.0, a.code)
Base.zero(::Type{Quantity}) = Quantity(0.0, "unit")
Base.float(a::Quantity) = val_canon(a)

function Base.show(io::IO, q::Quantity)
    print(io, round(q.val; sigdigits=6), " ", q.code)
end

"""
    CompoundUnit(raw, num, den, gas)

A *ratio* unit, as used by emission factors: `num` is the greenhouse-gas or
CO₂-equivalent mass, `den` the activity unit the factor refers to, `gas` the
species when the numerator is gas specific (`"CH4"`, `"N2O"`, `"CO2"`, else
`"CO2e"`):

    EFᵢ = 0.42 kgCO₂e/kWh  ⟶  num = kgCO₂e (:mass_co2e), den = kWh (:energy), gas = "CO2e"
    EFᵢ = 2.0  kgCH₄/t     ⟶  num = kg     (:mass),     den = t   (:mass),   gas = "CH4"
"""
struct CompoundUnit
    raw::String
    num::UnitDef
    den::Union{UnitDef,Nothing}
    gas::Union{String,Nothing}
end

const CO₂e_BY_PREFIX = Dict("g" => "gCO₂e", "kg" => "kgCO₂e", "t" => "tCO₂e",
                            "kt" => "ktCO₂e", "Mt" => "MtCO₂e")
const GAS_SPECIES = Dict("CH4" => "CH4", "CH₄" => "CH4", "CO2" => "CO2", "CO₂" => "CO2",
                         "N2O" => "N2O", "N₂O" => "N2O")
const CO₂e_SPELLINGS = ("CO2e", "CO₂e", "CO2eq", "CO2Eq", "CO2equivalent", "CO2e")

"""
    parse_compound("kgCO2e/kWh") -> CompoundUnit

Split an emission-factor unit into numerator (gas mass) and denominator
(activity), tolerating the spellings found in real factor libraries
(`kgCO2e/kWh`, `kg CO2 e / kWh`, `tCO2e per t`, `kgCH4/t`, `gN2O/MJ`).
"""
function parse_compound(code::AbstractString)
    s = replace(String(code), r"\s*per\s*"i => "/", r"\s+" => "")
    parts = split(s, '/')
    nraw = String(parts[1])
    gas, num = nothing, nothing
    m = match(r"^(g|kg|t|kt|Mt)?(CO2e|CO₂e|CO2eq|CO2Eq|CO2equiv|CO2|CO₂|CH4|CH₄|N2O|N₂O)$", nraw)
    if m !== nothing
        prefix = m.captures[1] === nothing ? "kg" : m.captures[1]
        species = m.captures[2]
        if species in CO₂e_SPELLINGS
            num, gas = UNITS[CO₂e_BY_PREFIX[prefix]], "CO2e"
        else
            prefix in ("g", "kg", "t") || throw(ArgumentError(
                "gas mass prefix «$prefix» unsupported in «$code»; use g, kg or t"))
            num, gas = UNITS[prefix], GAS_SPECIES[species]
        end
    else
        num = unit(nraw)                 # already a registered code, e.g. "kgCO₂e"
        gas = num.dim == :mass_co2e ? "CO2e" : nothing
    end
    den = length(parts) > 1 && !isempty(parts[2]) ? unit(parts[2]) : nothing
    CompoundUnit(String(code), num, den, gas)
end

"""
    adjust_ef(EFᵢ, ef_unit::CompoundUnit, activity_code) -> Float64

Rescale an emission factor so that its denominator matches the unit in which
the activity is measured, numerator unchanged:

    EF_adj = EFᵢ · f_activity / f_denominator
"""
function adjust_ef(EFᵢ::Real, ef_unit::CompoundUnit, activity_code::AbstractString)::Float64
    den = ef_unit.den
    den === nothing && throw(ArgumentError("emission factor «$(ef_unit.raw)» has no activity unit"))
    act = unit(activity_code)
    act.dim == den.dim || throw(DimensionMismatch(
        "activity «$activity_code» ($(act.dim)) does not match factor denominator $(den.code) ($(den.dim))"))
    Float64(EFᵢ) * act.factor / den.factor
end

"""
    emission(Q̇ᵢ::Quantity, EFᵢ::Quantity) -> Quantity

The core calculation of every scope,  **Eᵢ = Q̇ᵢ · EFᵢ**, returned in tCO₂e. A
`DimensionMismatch` is raised when the activity unit and the factor denominator
disagree — the most common error in hand-built inventories.
"""
function emission(Q̇ᵢ::Quantity, EFᵢ::Quantity)
    cu = parse_compound(EFᵢ.code)
    cu.den === nothing && throw(ArgumentError("«$(EFᵢ.code)» is not a per-activity factor"))
    cu.num.dim == :mass_co2e || throw(ArgumentError(
        "factor numerator «$(cu.num.code)» is $(cu.num.dim); use emission_gas() for gas-specific factors"))
    # the factor is rescaled to the unit the activity is measured in, and the
    # product is evaluated in exactly those units:  E = Q̇ᵢ · EF_adj (× 10⁻³ kg→t)
    E = Q̇ᵢ.val * adjust_ef(EFᵢ.val, cu, Q̇ᵢ.code) * cu.num.factor
    Quantity(E, "tCO₂e")
end

"""
    emission_gas(Q̇ᵢ, EFᵢ; gwp=GWP_AR6, fossil=true) -> Quantity

Same as [`emission`](@ref) for gas-specific factors (`kgCH₄/t`, `gN₂O/MJ`): the
mass of gas is multiplied by the global warming potential:

    E_CO₂e = GWP₁₀₀ · ( Q̇ᵢ · EFᵢ )
"""
function emission_gas(Q̇ᵢ::Quantity, EFᵢ::Quantity; gwp::GWPSet=GWP_AR6, fossil::Bool=true)
    cu = parse_compound(EFᵢ.code)
    cu.gas in (nothing, "CO2e") && return emission(Q̇ᵢ, EFᵢ)
    m_gas = Q̇ᵢ.val * adjust_ef(EFᵢ.val, cu, Q̇ᵢ.code) * cu.num.factor   # t of gas
    E = co2e(; CO₂=(cu.gas == "CO2" ? m_gas : 0.0), CH₄=(cu.gas == "CH4" ? m_gas : 0.0),
               N₂O=(cu.gas == "N2O" ? m_gas : 0.0), gwp=gwp, fossil=fossil)
    Quantity(E, "tCO₂e")
end

"""
    PriceIndex(base_year, cpi)

Consumer/producer price index used to deflate procurement spend to the base
year of an EEIO factor set:  s_spend = amount · CPI(base) / CPI(year).
"""
struct PriceIndex
    base_year::Int
    cpi::Dict{Int,Float64}
end

"Deflate a nominal amount from `year` to the base year of `idx`."
deflate(amount::Real, year::Int, idx::PriceIndex) =
    Float64(amount) * idx.cpi[idx.base_year] / get(idx.cpi, year, idx.cpi[idx.base_year])
