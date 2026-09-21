# ══════════════════════════════════════════════════════════════════════════════
#  ingest.jl — data parsing and normalisation
#
#  Raw activity data arrives from meters, ERP extracts, fuel invoices and
#  spreadsheets: mixed delimiters, units glued to the value ("1,234.5 MWh"),
#  several date conventions, multi-language headers, duplicated rows.
#
#  This module turns that into typed `ActivityRecord`s carrying
#      • a canonical unit per dimension (t, GJ, m³, kWh, USD2024, …),
#      • an explicit period 𝒫 = [τ₀, τ₁],
#      • a data-quality class DQ ∈ {1,2,3,4},
#      • full lineage: source file, SHA-256 of that file, line number,
#  and *quarantines* every row it cannot interpret, with a reason, instead of
#  silently dropping or coercing it.
# ══════════════════════════════════════════════════════════════════════════════

"Data-quality classes of the GHG-Protocol data hierarchy (1 = best)."
const DATA_QUALITY_CLASSES = Dict(
    1 => "supplier / meter specific, verified",
    2 => "site specific, measured, unverified",
    3 => "average data (industry or regional average)",
    4 => "proxy / spend-based estimate",
)

"""
    ActivityRecord

One normalised row of activity data — the unit of everything downstream. The
emission of the row is a single multiplication, `Eᵢ = Q̇ᵢ · EFᵢ`.
"""
Base.@kwdef struct ActivityRecord
    rec_id::String
    entity_id::String
    site_id::String
    source_type::String                     # stationary_combustion, electricity, purchased_goods, …
    scope::Symbol                           # :scope1 | :scope2 | :scope3
    cat::Union{Nothing,Int} = nothing       # Scope-3 category 1…15
    item::String = ""                       # fuel, refrigerant, product, service
    ts::Union{Nothing,DateTime} = nothing   # measurement timestamp, when available
    τ₀::Date                                # period start
    τ₁::Date                                # period end
    Q̇ᵢ::Quantity                            # activity exactly as parsed
    Q̇_canon::Float64                        # activity in the canonical unit
    unit_canon::String                      # canonical unit code
    amount::Union{Nothing,Float64} = nothing   # money, for spend-based methods
    currency::String = ""
    country::String = ""
    supplier::String = ""
    DQ::Int = 3                             # data-quality class
    meta::Dict{String,String} = Dict{String,String}()
    source::String = ""                     # file or connector identifier
    source_hash::String = ""                # SHA-256 of the raw file
    lineno::Int = 0
end

"One-line rendering: id, source type, item, activity in canonical units, period, lineage."
function Base.show(io::IO, r::ActivityRecord)
    print(io, r.rec_id, " │ ", r.source_type, " │ ", r.item, " │ ",
          round(r.Q̇_canon; sigdigits=6), " ", r.unit_canon, " │ ",
          r.τ₀, "→", r.τ₁, " │ DQ", r.DQ, " │ ", basename(r.source), ":", r.lineno)
end

"A row that could not be normalised, with the reason it was set aside."
struct QuarantinedRow
    source::String
    lineno::Int
    reason::String
    raw::Dict{String,String}
end

"""
    IngestReport

Result of one ingestion run: accepted records, quarantined rows and the
file-level lineage (path, SHA-256, row count, mtime).
"""
struct IngestReport
    records::Vector{ActivityRecord}
    quarantined::Vector{QuarantinedRow}
    files::Vector{Dict{String,Any}}
end

function Base.show(io::IO, r::IngestReport)
    println(io, "IngestReport: ", length(r.records), " records, ",
            length(r.quarantined), " quarantined, ", length(r.files), " file(s)")
    for f in r.files
        println(io, "   ", basename(string(f["path"])), "  sha256=", first(string(f["sha256"]), 12),
                "…  rows=", f["rows"])
    end
end

# ── scalar parsers ───────────────────────────────────────────────────────────
"""
    parse_number(s) -> Float64

Tolerant numeric parser for real exports: thousands separators as `,` or space,
decimal comma (`1.234,5`), currency symbols, `%`, non-breaking spaces.
"""
function parse_number(s::AbstractString)::Float64
    t = strip(replace(String(s), r"[\$€£]" => "", '\u00a0' => ' ', '\u202f' => ' '))
    isempty(t) && throw(ArgumentError("empty number"))
    t = replace(t, r"[\s_]" => "")
    t = replace(t, r"(?i)usd|eur|brl|cny|zar|r\$" => "")
    neg = startswith(t, "(") && endswith(t, ")")
    t = replace(t, r"[()%]" => "")
    if occursin(',', t) && occursin('.', t)
        t = findlast(',', t) > findlast('.', t) ? replace(replace(t, "." => ""), "," => ".") :
                                                 replace(t, "," => "")
    elseif occursin(',', t)
        t = length(split(t, ',')[end]) == 3 ? replace(t, "," => "") : replace(t, "," => ".")
    end
    neg ? -parse(Float64, t) : parse(Float64, t)
end

const DATE_FORMATS = ("yyyy-mm-dd", "dd/mm/yyyy", "dd.mm.yyyy", "yyyy/mm/dd",
                      "mm/dd/yyyy", "dd-mm-yyyy", "yyyy-mm-ddTHH:MM:SS", "dd-mon-yyyy")

"""
    parse_date(s; dayfirst=true) -> Date

Parses the date conventions found in the sample data. `dayfirst` resolves the
`01/02/2024` ambiguity explicitly instead of guessing silently.
"""
function parse_date(s::AbstractString; dayfirst::Bool=true)
    t = strip(String(s))
    # a date must name a year: without this check "01" would parse as year 1
    (isempty(t) || !occursin(r"\d{4}", t)) && throw(ArgumentError("unrecognised date «$s»"))
    for f in (dayfirst ? DATE_FORMATS : reverse(DATE_FORMATS))
        d = try
            DateTime(t, Dates.DateFormat(f))
        catch
            continue
        end
        return Date(d)
    end
    throw(ArgumentError("unrecognised date «$s»"))
end

"""
    parse_period(s) -> (Date, Date)

Accepts `2024-Q1`, `2024-03`, `2024`, `01.01.2024-31.12.2024` and
`2024-03-01/2024-03-31`: the period labels that appear on invoices and monthly
meter reports.
"""
function parse_period(s::AbstractString)
    t = strip(String(s))
    for sep in (" - ", " – ", "-", "/")
        parts = split(t, sep)
        if length(parts) == 2 && all(p -> occursin(r"\d{4}", p), parts)
            a = try
                parse_date(parts[1])
            catch
                nothing
            end
            b = try
                parse_date(parts[2])
            catch
                nothing
            end
            (a !== nothing && b !== nothing) && return (min(a, b), max(a, b))
        end
    end
    m = match(r"^(\d{4})-?Q([1-4])$", uppercase(t))
    if m !== nothing
        y, q = parse(Int, m.captures[1]), parse(Int, m.captures[2])
        d = Date(y, 3q - 2, 1)
        return (d, lastdayofquarter(d))
    end
    m = match(r"^(\d{4})-(\d{1,2})$", t)
    if m !== nothing
        d = Date(parse(Int, m.captures[1]), parse(Int, m.captures[2]), 1)
        return (d, lastdayofmonth(d))
    end
    occursin(r"^\d{4}$", t) && return (Date(parse(Int, t), 1, 1), Date(parse(Int, t), 12, 31))
    throw(ArgumentError("unrecognised period «$s»"))
end

"Fold a column heading into a lookup key: lower case, no accents, no separators."
function header_key(name::AbstractString)
    s = lowercase(strip(String(name)))
    s = replace(s, r"[áàâäã]" => "a", r"[éèêë]" => "e", r"[íìîï]" => "i", r"[óòôöõ]" => "o",
                r"[úùûü]" => "u", "ç" => "c", "ñ" => "n", "ß" => "ss")
    replace(s, r"[^a-z0-9]+" => "_")
end

"""
    DEFAULT_FIELD_MAP

Alias table used to recognise the canonical fields of an activity file in
arbitrary exports (English, German, Portuguese, Spanish, French and Dutch
spellings plus common ERP abbreviations). Extend it per deployment with
`ingest_files(paths; field_map=merge(DEFAULT_FIELD_MAP, my_map))`.
"""
const DEFAULT_FIELD_MAP = Dict{String,String}(
    # identity and location
    "entity" => "entity_id", "entity_id" => "entity_id", "company" => "entity_id",
    "empresa" => "entity_id", "unternehmen" => "entity_id", "legal_entity" => "entity_id",
    "site" => "site_id", "site_id" => "site_id", "plant" => "site_id", "werk" => "site_id",
    "unidade" => "site_id", "facility" => "site_id", "installation" => "site_id",
    "country" => "country", "pais" => "country", "land" => "country", "iso" => "country",
    "iso_country" => "country", "region" => "country",
    "supplier" => "supplier", "vendor" => "supplier", "fornecedor" => "supplier",
    "lieferant" => "supplier", "proveedor" => "supplier",
    # what is measured
    "source_type" => "source_type", "category" => "source_type", "activity_type" => "source_type",
    "tipo" => "source_type", "emission_source" => "source_type", "source" => "source_type",
    "kategorie" => "source_type", "kategoria" => "source_type", "art" => "source_type",
    "fuel" => "item", "fuel_type" => "item", "item" => "item", "material" => "item",
    "combustivel" => "item", "kraftstoff" => "item", "brennstoff" => "item",
    "product" => "item", "producto" => "item", "refrigerant" => "item",
    "route" => "item", "tratamento" => "item", "entsorgung" => "item", "mode" => "item",
    "description" => "item", "descricao" => "item", "bezeichnung" => "item", "descripcion" => "item",
    # how much
    "quantity" => "quantity", "qty" => "quantity", "amount" => "quantity", "value" => "quantity",
    "consumption" => "quantity", "verbrauch" => "quantity", "consumo" => "quantity",
    "menge" => "quantity", "volume" => "quantity", "energy" => "quantity",
    "quantity_unit" => "unit", "unit" => "unit", "uom" => "unit", "unidad" => "unit",
    "einheit" => "unit", "units" => "unit",
    "mass" => "quantity", "massa" => "quantity", "masse" => "quantity", "weight" => "quantity",
    # when
    "period" => "period", "month" => "period", "quarter" => "period", "year" => "period",
    "mes" => "period", "ano" => "period", "zeitraum" => "period", "periode" => "period",
    "date" => "ts", "timestamp" => "ts",
    "reading_date" => "ts", "fecha" => "ts", "datum" => "ts", "data" => "ts",
    "period_start" => "τ₀", "start" => "τ₀", "from" => "τ₀",
    "period_end" => "τ₁", "end" => "τ₁", "to" => "τ₁",
    # money
    "spend" => "amount", "cost" => "amount", "expenditure" => "amount", "valor" => "amount",
    "betrag" => "amount", "importe" => "amount", "invoice_value" => "amount", "price" => "amount",
    "currency" => "currency", "moeda" => "currency", "ccy" => "currency",
    # quality, provenance and free text
    "dq" => "DQ", "data_quality" => "DQ", "quality" => "DQ", "datenqualitat" => "DQ",
    "datenqualitaet" => "DQ", "qualidade" => "DQ", "qualitaet" => "DQ", "data_source" => "provenance",
    "method" => "method", "metodo" => "method", "notes" => "note", "comment" => "note",
    "nota" => "note", "bemerkung" => "note", "anmerkung" => "note", "meter_id" => "meter_id",
    "invoice_id" => "invoice_id", "document" => "document", "ticket" => "ticket",
)

"Map the columns of one raw row onto canonical field names."
function map_fields(row::AbstractDict, field_map::AbstractDict=DEFAULT_FIELD_MAP)
    out = Dict{String,String}()
    for (k, v) in row
        canon = get(field_map, header_key(String(k)), header_key(String(k)))
        out[canon] = ismissing(v) ? "" : strip(string(v))
    end
    out
end

# ── source-type rules: free text → scope, expected dimension, default quality ─
"""
    SourceRule(key, scope, cat, dim, unit_hint, DQ, label)

The machine-readable part of the reporting policy: which scope and category an
activity belongs to, which dimension its quantity must have, which unit is
assumed when the file states none, and the data-quality class that this source
type deserves *by construction* (a supplier meter reading is class 2, a
spend-based estimate class 4).
"""
struct SourceRule
    key::String
    scope::Symbol
    cat::Union{Nothing,Int}
    dim::Symbol
    unit_hint::String
    DQ::Int
    label::String
end

const SOURCE_RULES = Dict{String,SourceRule}()
register_source!(key, scope, cat, dim, unit_hint, DQ, label) =
    (SOURCE_RULES[key] = SourceRule(key, scope, cat, dim, unit_hint, DQ, label))

function _register_sources!()
    empty!(SOURCE_RULES)
    # ── Scope 1: stationary combustion (energy input) ────────────────────────
    for (k, lbl) in [("natural_gas", "natural gas boiler / kiln"),
                     ("fuel_oil", "fuel oil combustion"),
                     ("diesel_stationary", "diesel generator (stationary)"),
                     ("lpg", "liquefied petroleum gas"),
                     ("coal", "coal / petcoke combustion"),
                     ("biomass", "biomass / bagasse combustion"),
                     ("blast_furnace_gas", "process off-gas combustion")]
        register_source!(k, :scope1, nothing, :energy, "GJ", 2, lbl)
    end
    # ── Scope 1: mobile combustion ───────────────────────────────────────────
    register_source!("mobile_combustion", :scope1, nothing, :energy, "GJ", 2, "fleet / off-road combustion")
    register_source!("diesel_fleet", :scope1, nothing, :volume, "L", 2, "diesel road fleet")
    register_source!("gasoline_fleet", :scope1, nothing, :volume, "L", 2, "gasoline road fleet")
    register_source!("aviation_fuel", :scope1, nothing, :mass, "t", 2, "jet fuel (owned aircraft)")
    register_source!("marine_fuel", :scope1, nothing, :mass, "t", 2, "bunker fuel (owned vessel)")
    # ── Scope 1: fugitive ────────────────────────────────────────────────────
    register_source!("refrigerant", :scope1, nothing, :mass, "kg", 2, "refrigerant leakage (mass balance)")
    register_source!("ch4_vent", :scope1, nothing, :volume, "m³(n)", 3, "vented / released methane")
    register_source!("ldar", :scope1, nothing, :count, "component", 4, "LDAR leak-rate estimate")
    register_source!("compressor_seal", :scope1, nothing, :mass, "kg", 3, "compressor seal losses")
    # ── Scope 1: flaring and process ─────────────────────────────────────────
    register_source!("flaring", :scope1, nothing, :volume, "m³(n)", 2, "flare gas sent to flare")
    register_source!("process_cement", :scope1, nothing, :mass, "t", 2, "clinker production (process CO₂)")
    register_source!("process_lime", :scope1, nothing, :mass, "t", 2, "lime production")
    register_source!("process_nitric_acid", :scope1, nothing, :mass, "t", 2, "nitric acid (N₂O)")
    register_source!("process_ammonia", :scope1, nothing, :mass, "t", 2, "ammonia synthesis")
    register_source!("process_aluminium", :scope1, nothing, :mass, "t", 2, "anode consumption / PFC")
    # ── Scope 2: purchased energy ────────────────────────────────────────────
    register_source!("electricity", :scope2, nothing, :energy, "MWh", 2, "purchased electricity")
    register_source!("ppa_electricity", :scope2, nothing, :energy, "MWh", 1, "PPA-covered electricity (market based)")
    register_source!("steam", :scope2, nothing, :energy, "GJ", 2, "purchased steam")
    register_source!("heat", :scope2, nothing, :energy, "GJ", 2, "purchased heat")
    register_source!("cooling", :scope2, nothing, :energy, "GJ", 2, "purchased cooling")
    # ── Scope 3, categories 1…15 ────────────────────────────────────────────
    register_source!("purchased_goods_spend", :scope3, 1, :currency, "USD2024", 4, "cat 1 – spend-based goods & services")
    register_source!("purchased_goods_mass", :scope3, 1, :mass, "t", 2, "cat 1 – mass-based goods")
    register_source!("capital_goods", :scope3, 2, :currency, "USD2024", 4, "cat 2 – capital goods (spend)")
    register_source!("fuel_energy_upstream", :scope3, 3, :energy, "GJ", 3, "cat 3 – upstream fuel & energy (WTT)")
    register_source!("upstream_transport", :scope3, 4, :transport_work, "t·km", 3, "cat 4 – upstream transport (t·km)")
    register_source!("waste", :scope3, 5, :mass, "t", 3, "cat 5 – waste treatment")
    register_source!("business_travel_air", :scope3, 6, :transport_work, "p·km", 3, "cat 6 – business travel, air (p·km)")
    register_source!("business_travel_land", :scope3, 6, :transport_work, "p·km", 3, "cat 6 – business travel, land (p·km)")
    register_source!("employee_commuting", :scope3, 7, :transport_work, "p·km", 4, "cat 7 – employee commuting (p·km)")
    register_source!("upstream_leased", :scope3, 8, :currency, "USD2024", 4, "cat 8 – upstream leased assets")
    register_source!("downstream_transport", :scope3, 9, :transport_work, "t·km", 3, "cat 9 – downstream transport (t·km)")
    register_source!("processing_sold", :scope3, 10, :mass, "t", 4, "cat 10 – processing of sold products")
    register_source!("use_phase", :scope3, 11, :energy, "GJ", 3, "cat 11 – use of sold products")
    register_source!("end_of_life", :scope3, 12, :mass, "t", 4, "cat 12 – end-of-life of sold products")
    register_source!("downstream_leased", :scope3, 13, :currency, "USD2024", 4, "cat 13 – downstream leased assets")
    register_source!("franchises", :scope3, 14, :currency, "USD2024", 4, "cat 14 – franchises")
    register_source!("investments", :scope3, 15, :currency, "USD2024", 4, "cat 15 – investments")
    return SOURCE_RULES
end

_register_sources!()

"Natural-language and multi-language aliases for the source types."
const SOURCE_ALIASES = Dict{String,String}(
    "erdgas" => "natural_gas", "gas_kessel" => "natural_gas", "gaz_naturel" => "natural_gas",
    "gas_natural" => "natural_gas", "heizol" => "fuel_oil", "oleo" => "fuel_oil",
    "carvao" => "coal", "kohle" => "coal", "lenha" => "biomass", "bagaco" => "biomass",
    "electricidade" => "electricity", "electricidad" => "electricity", "strom" => "electricity",
    "energia_eletrica" => "electricity", "vapor" => "steam", "dampf" => "steam",
    "calor" => "heat", "waerme" => "heat", "frio" => "cooling", "kalt" => "cooling",
    "frota" => "diesel_fleet", "flotte" => "diesel_fleet", "fuel_fleet" => "diesel_fleet",
    "refrigerante" => "refrigerant", "kaeltemittel" => "refrigerant", "r134a" => "refrigerant",
    "r410a" => "refrigerant", "r744" => "refrigerant", "r32" => "refrigerant",
    "ch4_leak" => "ch4_vent", "methan" => "ch4_vent", "metano" => "ch4_vent",
    "flare" => "flaring", "fackel" => "flaring", "queima" => "flaring",
    "clinker" => "process_cement", "clinquer" => "process_cement", "zement" => "process_cement",
    "kalk" => "process_lime", "cal" => "process_lime", "salpetersaeure" => "process_nitric_acid",
    "salpetersaure" => "process_nitric_acid", "acido_nitrico" => "process_nitric_acid",
    "amoniaco" => "process_ammonia", "ammoniak" => "process_ammonia",
    "residuo" => "waste", "abfall" => "waste", "aterro" => "waste", "dechets" => "waste",
    "viagem_negocios" => "business_travel_air", "flug" => "business_travel_air",
    "commuting" => "employee_commuting", "pendeln" => "employee_commuting",
    "goods_services" => "purchased_goods_spend", "bienes" => "purchased_goods_spend",
    "bens_e_servicos" => "purchased_goods_spend", "dienstleistung" => "purchased_goods_spend",
    "transport_upstream" => "upstream_transport", "fracht" => "upstream_transport",
    "transporte" => "downstream_transport", "use_phase_product" => "use_phase",
    "investimento" => "investments",
)

"""
    lookup_source(text) -> Union{SourceRule,Nothing}

Resolve a free-text source type ("Erdgas Kessel 3", "Purchased electricity",
"clinker") to a rule: exact key first, then substring match, then the alias
table.
"""
function lookup_source(text::AbstractString)
    k = header_key(String(text))
    isempty(k) && return nothing
    haskey(SOURCE_RULES, k) && return SOURCE_RULES[k]
    for (key, rule) in SOURCE_RULES
        length(k) > 3 && (occursin(key, k) || occursin(k, key)) && return rule
    end
    for (alias_, key) in SOURCE_ALIASES
        occursin(alias_, k) && return SOURCE_RULES[key]
    end
    nothing
end

"ISO-3166 alpha-2 → alpha-3 for the countries of the sample portfolio."
const ISO_ALPHA3 = Dict("BR" => "BRA", "DE" => "DEU", "ZA" => "ZAF", "US" => "USA",
    "CN" => "CHN", "NL" => "NLD", "ES" => "ESP", "FR" => "FRA", "PT" => "PRT",
    "GB" => "GBR", "UK" => "GBR", "IN" => "IND", "MX" => "MEX", "CL" => "CHL",
    "AR" => "ARG", "PL" => "POL", "IT" => "ITA", "NO" => "NOR", "SE" => "SWE")

"Normalise a country label to ISO-3166 alpha-3."
function iso_country(text::AbstractString)
    c = uppercase(strip(String(text)))
    isempty(c) && return ""
    length(c) == 3 && return c
    length(c) == 2 && return get(ISO_ALPHA3, c, c)
    haskey(NAME_TO_ALPHA3, header_key(c)) && return NAME_TO_ALPHA3[header_key(c)]
    c
end

const NAME_TO_ALPHA3 = Dict("brazil" => "BRA", "brasil" => "BRA", "germany" => "DEU",
    "deutschland" => "DEU", "southafrica" => "ZAF", "unitedstates" => "USA",
    "usa" => "USA", "china" => "CHN", "netherlands" => "NLD", "spain" => "ESP",
    "france" => "FRA", "portugal" => "PRT", "unitedkingdom" => "GBR", "poland" => "POL")

# ── the normalisation pipeline ───────────────────────────────────────────────
"""
    read_delimited(path; delim=nothing) -> NamedTuple

Read a delimited text export into raw string rows. The delimiter is sniffed from
the header line when not given (`,` `;` tab `|` are the usual suspects).
"""
function read_delimited(path::AbstractString; delim::Union{Nothing,Char}=nothing)
    d = delim
    if d === nothing
        line = first(eachline(path))
        cands = [',', ';', '\t', '|']
        d = cands[argmax([count(==(c), line) for c in cands])]
    end
    tbl = CSV.File(path; delim=d, header=1, types=String, missingstring=String[],
                   stripwhitespace=true)
    rows = [Dict{String,String}(string(k) => (v === missing ? "" : string(v)) for (k, v) in pairs(r))
            for r in tbl]
    (rows=rows, delim=d)
end

"""
    normalize_row(raw; field_map, dayfirst, source, source_hash, lineno) -> ActivityRecord

The heart of the parsing layer. It
  1. maps the columns through `field_map`,
  2. resolves the free-text source type to a [`SourceRule`](@ref) (scope, cat, dimension),
  3. splits value and unit ("1,234.5 MWh" or value column + unit column),
  4. checks the *dimension* of the quantity against the rule and converts to the
     canonical unit,
  5. resolves the period label to 𝒫 = [τ₀, τ₁],
  6. records data-quality class, country, currency, money and lineage.

Any failure raises with a human-readable reason; the caller quarantines the row.
"""
function normalize_row(raw::AbstractDict;
                       field_map::AbstractDict=DEFAULT_FIELD_MAP,
                       dayfirst::Bool=true,
                       source::AbstractString="",
                       source_hash::AbstractString="",
                       lineno::Integer=0,
                       entity_default::AbstractString="")
    f = map_fields(raw, field_map)
    st_text = coalesce_nonempty(get(f, "source_type", ""), get(f, "item", ""))
    rule = lookup_source(st_text)
    rule === nothing && throw(ArgumentError("unknown source type «$st_text» (no matching SourceRule)"))

    # ── value and unit ───────────────────────────────────────────────────────
    qty_text = strip(get(f, "quantity", ""))
    unit_text = strip(get(f, "unit", ""))
    cur_text = uppercase(strip(get(f, "currency", "")))
    isempty(qty_text) && throw(ArgumentError("missing quantity"))
    Q̇ᵢ = if !isempty(unit_text)
        Quantity(parse_number(qty_text), unit(unit_text).code)
    elseif !isempty(cur_text)
        # spend-based rows carry money in the quantity column, currency in its own
        Quantity(parse_number(qty_text), unit(cur_text).code)
    else
        parse_quantity(qty_text)                      # unit glued to the value
    end
    dimension(Q̇ᵢ) == rule.dim || throw(ArgumentError(
        "quantity «$(Q̇ᵢ.code)» has dimension $(dimension(Q̇ᵢ)) but source type «$(rule.key)» expects $(rule.dim)"))

    # ── period ───────────────────────────────────────────────────────────────
    τ₀, τ₁ = nothing, nothing
    if !isempty(get(f, "τ₀", "")) && !isempty(get(f, "τ₁", ""))
        τ₀, τ₁ = parse_date(get(f, "τ₀", ""); dayfirst), parse_date(get(f, "τ₁", ""); dayfirst)
    elseif !isempty(get(f, "period", ""))
        τ₀, τ₁ = parse_period(get(f, "period", ""))
    elseif !isempty(get(f, "ts", ""))
        τ₀ = parse_date(get(f, "ts", ""); dayfirst)
        τ₁ = lastdayofmonth(τ₀)
    end
    τ₀ === nothing && throw(ArgumentError("no usable period or date column"))

    # ── money, geography, quality ────────────────────────────────────────────
    amount = isempty(get(f, "amount", "")) ? nothing : try
        parse_number(get(f, "amount", ""))
    catch
        nothing
    end
    currency = uppercase(strip(get(f, "currency", "")))
    if rule.dim == :currency && (amount === nothing || amount <= 0)
        amount = Q̇ᵢ.val                       # spend given in the quantity column
        Q̇ᵢ = Quantity(amount, currency == "" ? "USD2024" : currency)
    end
    country = iso_country(get(f, "country", ""))
    dq_text = strip(get(f, "DQ", ""))
    DQ = isempty(dq_text) ? rule.DQ : clamp(try
        round(Int, parse_number(dq_text))
    catch
        rule.DQ
    end, 1, 4)

    ActivityRecord(
        rec_id=string(isempty(source) ? "row" : basename(source), ":", lineno),
        entity_id=coalesce_nonempty(get(f, "entity_id", ""), entity_default),
        site_id=get(f, "site_id", ""),
        source_type=rule.key,
        scope=rule.scope,
        cat=rule.cat,
        item=coalesce_nonempty(get(f, "item", ""), rule.label),
        ts=nothing,
        τ₀=τ₀, τ₁=τ₁,
        Q̇ᵢ=Q̇ᵢ,
        Q̇_canon=val_canon(Q̇ᵢ),
        unit_canon=CANONICAL[dimension(Q̇ᵢ)],
        amount=amount, currency=currency, country=country,
        supplier=get(f, "supplier", ""),
        DQ=DQ,
        meta=Dict{String,String}("method" => get(f, "method", ""), "note" => redact(get(f, "note", "")),
                                 "meter_id" => get(f, "meter_id", ""),
                                 "document" => get(f, "document", ""),
                                 "instrument" => get(f, "instrument", ""),
                                 "equity_share" => get(f, "equity_share", "")),
        source=String(source), source_hash=String(source_hash), lineno=Int(lineno))
end

"First non-empty string of the arguments (used for coalescing optional fields)."
coalesce_nonempty(a::AbstractString, b::AbstractString) = isempty(strip(a)) ? String(b) : String(a)

# ── ingestion driver: files → records, with quarantine, dedup and lineage ────
"Natural key used to detect duplicates across files and sync runs."
natural_key(r::ActivityRecord) = (r.entity_id, r.site_id, r.source_type, r.item,
    string(r.τ₀), string(r.τ₁), round(r.Q̇_canon; digits=6), r.unit_canon, r.amount)

"""
    deduplicate(records) -> (unique, duplicates)

Drop rows repeating the same natural key — the same meter reading delivered twice
by two exports, or re-delivered by an idempotent connector.
"""
function deduplicate(records::AbstractVector{ActivityRecord})
    seen = Set{Any}()
    uniq, dups = ActivityRecord[], ActivityRecord[]
    for r in records
        k = natural_key(r)
        if k in seen
            push!(dups, r)
        else
            push!(seen, k)
            push!(uniq, r)
        end
    end
    (uniq, dups)
end

"""
    ingest_files(paths; kwargs...) -> IngestReport

Ingest one or many delimited exports. Every file is hashed (SHA-256) and, when a
`ledger` and `actor` are supplied, the ingestion itself is written to the audit
trail — so the *provenance* of every figure is provable, not merely asserted.
"""
function ingest_files(paths::AbstractVector;
                      field_map::AbstractDict=DEFAULT_FIELD_MAP,
                      dayfirst::Bool=true,
                      entity_default::AbstractString="",
                      ledger::Union{Nothing,AuditLedger}=nothing,
                      actor::Union{Nothing,Actor}=nothing,
                      delim::Union{Nothing,Char}=nothing,
                      context::AbstractString="periodic ingestion")
    records, quarantined, files = ActivityRecord[], QuarantinedRow[], Dict{String,Any}[]
    for p in paths
        isfile(p) || throw(ArgumentError("input file not found: $p"))
        h = sha256_file(p)
        rd = read_delimited(p; delim=delim)
        for (i, raw) in enumerate(rd.rows)
            try
                push!(records, normalize_row(raw; field_map=field_map, dayfirst=dayfirst,
                    source=p, source_hash=h, lineno=i + 1, entity_default=entity_default))
            catch e
                push!(quarantined, QuarantinedRow(p, i + 1,
                    e isa ArgumentError ? e.msg : sprint(showerror, e), raw))
            end
        end
        push!(files, Dict{String,Any}("path" => p, "sha256" => h, "rows" => length(rd.rows),
                                      "delim" => string(rd.delim)))
        if ledger !== nothing && actor !== nothing
            record!(ledger, actor, "ingest", basename(p); why=context, source_hash=h,
                    after="$(length(records)) records accepted, $(length(quarantined)) quarantined")
        end
    end
    uniq, dups = deduplicate(records)
    for d in dups
        push!(quarantined, QuarantinedRow(d.source, d.lineno,
            "duplicate of an earlier row (same natural key)", Dict("rec_id" => d.rec_id)))
    end
    IngestReport(uniq, quarantined, files)
end

ingest_file(path::AbstractString; kwargs...) = ingest_files([path]; kwargs...)

"""
    validate_records(records; year=nothing) -> Vector{NamedTuple}

Reporting controls applied to the normalised data *before* any calculation:
non-positive activity, missing entity or site, periods outside the reporting
year, inverted periods, proxy-quality rows and implausible magnitudes.
"""
function validate_records(records::AbstractVector{ActivityRecord}; year::Union{Nothing,Int}=nothing)
    issues = NamedTuple{(:rec_id, :severity, :message),Tuple{String,Symbol,String}}[]
    for r in records
        r.Q̇_canon <= 0 && push!(issues, (r.rec_id, :error, "non-positive activity $(r.Q̇_canon) $(r.unit_canon)"))
        isempty(r.entity_id) && push!(issues, (r.rec_id, :error, "missing entity_id (master-data gap)"))
        isempty(r.site_id) && push!(issues, (r.rec_id, :warning, "missing site_id"))
        r.τ₁ < r.τ₀ && push!(issues, (r.rec_id, :error, "inverted period $(r.τ₀) > $(r.τ₁)"))
        if year !== nothing && !(Dates.year(r.τ₀) == year && Dates.year(r.τ₁) == year)
            push!(issues, (r.rec_id, :error, "period $(r.τ₀)→$(r.τ₁) outside reporting year $year"))
        end
        r.DQ == 4 && push!(issues, (r.rec_id, :info, "proxy/spend-based estimate (DQ 4) — refine with supplier data"))
        abs(r.Q̇_canon) > 1e7 && push!(issues, (r.rec_id, :warning, "implausible magnitude $(r.Q̇_canon) $(r.unit_canon)"))
    end
    issues
end

"Records as a DataFrame, for notebooks, exports and the report layer."
function records_dataframe(records::AbstractVector{ActivityRecord})
    DataFrame(
        rec_id=[r.rec_id for r in records],
        entity=[r.entity_id for r in records],
        site=[r.site_id for r in records],
        source_type=[r.source_type for r in records],
        scope=[string(r.scope) for r in records],
        cat=[r.cat === nothing ? missing : r.cat for r in records],
        item=[r.item for r in records],
        τ₀=[r.τ₀ for r in records],
        τ₁=[r.τ₁ for r in records],
        value=[r.Q̇ᵢ.val for r in records],
        unit=[r.Q̇ᵢ.code for r in records],
        Q̇_canon=[r.Q̇_canon for r in records],
        unit_canon=[r.unit_canon for r in records],
        amount=[r.amount === nothing ? missing : r.amount for r in records],
        currency=[r.currency for r in records],
        country=[r.country for r in records],
        DQ=[r.DQ for r in records],
        source=[basename(r.source) for r in records],
        source_hash=[first(r.source_hash, 12) for r in records],
    )
end

"Quarantined rows as a DataFrame."
quarantine_dataframe(rows::AbstractVector{QuarantinedRow}) =
    DataFrame(source=[basename(q.source) for q in rows], lineno=[q.lineno for q in rows],
              reason=[q.reason for q in rows])

"Compact per-scope / per-source-type summary of an ingestion run."
function ingest_summary(rep::IngestReport)
    isempty(rep.records) && return DataFrame()
    df = records_dataframe(rep.records)
    grouped = combine(groupby(df, [:scope, :source_type]),
        :Q̇_canon => sum => :activity, nrow => :n)
    sort!(grouped, [:scope, :source_type])
    grouped
end
