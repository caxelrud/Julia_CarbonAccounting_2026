# ══════════════════════════════════════════════════════════════════════════════
#  notation.jl — the symbolic dictionary  (part 1/3)
#
#  Design rule of this code base: *every* quantity that appears in an equation
#  gets a variable name that IS the mathematical symbol (Unicode), so that the
#  code can be read next to the published formula:
#
#        E_Scope1 = Σᵢ ( Q̇ᵢ · EFᵢ )      ⟶   E_Scope1 = Σᵢ(i -> Q̇ᵢ[i] * EFᵢ[i], 1:n)
#
#  This file is the single source of truth for the symbols: it stores the
#  Unicode identifier, the LaTeX rendering, the physical meaning and the unit.
#  Notebook 00 and the generated PDF print this table as the "notation legend".
# ══════════════════════════════════════════════════════════════════════════════

"""
    SymbolDef(symbol, latex, meaning, unit, area)

One row of the notation legend: `symbol` is the literal Unicode identifier used
in the Julia source, `latex` how it is typeset, `meaning` the physical or
statistical definition, `unit` its measurement unit and `area` the engine it
belongs to.
"""
struct SymbolDef
    symbol::String
    latex::String
    meaning::String
    unit::String
    area::String
end

function Base.show(io::IO, s::SymbolDef)
    print(io, rpad(s.symbol, 14), " ", rpad(s.unit, 16), " ", s.meaning)
end

# ── physical constants used by the stoichiometric engines ────────────────────
const M_CO₂ = 44.0095   # g/mol, molar mass of carbon dioxide
const M_CH₄ = 16.043    # g/mol, molar mass of methane
const M_N₂O = 44.0128   # g/mol, molar mass of nitrous oxide
const M_C   = 12.011    # g/mol, molar mass of carbon
const ρ_CH₄ = 0.7168     # kg/m³ at 0 °C, 1.01325 bar (normal conditions)
const ρ_CO₂ = 1.9767     # kg/m³ at 0 °C, 1.01325 bar

"""
    GWPSet

100-year global-warming-potential set, used to express every greenhouse gas in
CO₂-equivalent mass:

    E_CO₂e = E_CO₂ + GWP_CH₄ · E_CH₄ + GWP_N₂O · E_N₂O

`CH₄_fossil` follows the IPCC distinction between fossil and biogenic methane.
"""
struct GWPSet
    name::String
    year::Int
    CH₄_fossil::Float64
    CH₄_biogenic::Float64
    N₂O::Float64
    source::String
end

const GWP_AR6 = GWPSet("IPCC AR6, GWP₁₀₀", 2021, 29.8, 27.0, 273.0,
    "IPCC AR6 WGI Table 7.15 (fossil/biogenic CH₄, N₂O with climate feedbacks)")
const GWP_AR5 = GWPSet("IPCC AR5, GWP₁₀₀", 2014, 30.0, 28.0, 265.0,
    "IPCC AR5 WGI Table 8.7")
const GWP_AR4 = GWPSet("IPCC AR4, GWP₁₀₀", 2007, 25.0, 25.0, 298.0,
    "IPCC AR4 WGI Table 2.14")

"""
    co2e(; CO₂=0.0, CH₄=0.0, N₂O=0.0, gwp=GWP_AR6, fossil=true) -> Float64

Convert a three-gas emission vector into CO₂-equivalent mass, in the same mass
unit as the inputs, applying the methane-photochemistry correction for fossil
versus biogenic CH₄.
"""
function co2e(; CO₂::Real=0.0, CH₄::Real=0.0, N₂O::Real=0.0,
              gwp::GWPSet=GWP_AR6, fossil::Bool=true)
    GWP_CH₄ = fossil ? gwp.CH₄_fossil : gwp.CH₄_biogenic
    CO₂ + GWP_CH₄ * CH₄ + gwp.N₂O * N₂O
end

"""
    Σᵢ(f, xs)
    Σᵢ(xs)

Explicit summation operator, written as the mathematical symbol. Used wherever
the aggregation *is* part of a published formula, so that the code carries the
equation.
"""
function Σᵢ(f, xs)
    isempty(xs) && return 0.0          # an empty sum is zero, not an error
    sum(f, xs)
end
Σᵢ(xs) = isempty(xs) ? 0.0 : sum(xs)

"""
    Vᵢⱼ(a⃗, b⃗)

Outer product `Vᵢⱼ = aᵢ·bⱼ`, the elementary step of a neural-network layer
(`z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾`) and of covariance updates in the anomaly engine.
The inner product itself is written with Julia's `⋅` operator:

        zⱼ = w⃗ⱼ ⋅ a⃗ + bⱼ
"""
Vᵢⱼ(a⃗, b⃗) = a⃗ * transpose(b⃗)

# ── the legend (filled in parts 2/3) ─────────────────────────────────────────
const NOTATION = SymbolDef[
    # ── boundary, period, master data ────────────────────────────────────────
    SymbolDef("Entity", "\\mathcal{E}", "legal entity / installation inside the reporting boundary", "—", "boundary"),
    SymbolDef("𝒜", "\\mathcal{A}", "set of activities attributed to Entity inside period 𝒫", "—", "boundary"),
    SymbolDef("𝒫", "\\mathcal{P}", "reporting period, 𝒫 = [τ₀, τ₁]; one calendar year", "—", "boundary"),
    SymbolDef("©ᵢ", "\\copyright_i", "consolidation share of activity i (equity or control), 0 ≤ © ≤ 1", "—", "boundary"),
    SymbolDef("𝔅", "\\mathfrak{B}", "approach ∈ {equity share, financial control, operational control}", "—", "boundary"),
    # ── activity data ────────────────────────────────────────────────────────
    SymbolDef("Q̇ᵢ", "\\dot{Q}_i", "activity rate of source i: energy or mass flow per period", "GJ, t, kWh", "activity"),
    SymbolDef("ṁᵢ", "\\dot{m}_i", "mass flow of fuel i  (ṁ = ṅ · M: mole flow × molar mass)", "t/𝒫", "activity"),
    SymbolDef("ṅᵢ", "\\dot{n}_i", "molar flow of species i", "kmol/𝒫", "activity"),
    SymbolDef("V̇ᵢ", "\\dot{V}_i", "gas volumetric flow, normal conditions (0 °C, 1.01325 bar)", "m³(n)/𝒫", "activity"),
    SymbolDef("OFᵢ", "OF_i", "oxidation factor: carbon actually oxidised / carbon input (IPCC)", "—", "activity"),
    SymbolDef("η_b", "\\eta_b", "boiler or combustion efficiency", "—", "activity"),
    SymbolDef("DRE", "DRE", "destruction & removal efficiency of a flare / abatement device", "—", "activity"),
    SymbolDef("F_i", "F_i", "load or utilisation factor of equipment i", "—", "activity"),
    SymbolDef("h_op", "h_{op}", "operating hours of a source inside 𝒫", "h/𝒫", "activity"),
    SymbolDef("LHVᵢ", "LHV_i", "lower heating value of fuel i", "GJ/t", "activity"),
    SymbolDef("CCᵢ", "CC_i", "carbon content of fuel i", "tC/t", "activity"),
    SymbolDef("ΔPᵢ", "\\Delta P_i", "leak-driving pressure differential of source i", "bar", "activity"),
    # ── emission factors ─────────────────────────────────────────────────────
    SymbolDef("EFᵢ", "EF_i", "emission factor of source i (gas specific, per unit of activity)", "kg gas/unit", "factors"),
    SymbolDef("EF_CO₂", "EF_{\\mathrm{CO_2}}", "CO₂ factor:  CC · (44/12) · OF", "kgCO₂/t", "factors"),
    SymbolDef("EF_CH₄", "EF_{\\mathrm{CH_4}}", "CH₄ factor per unit of activity", "kgCH₄/t", "factors"),
    SymbolDef("EF_N₂O", "EF_{\\mathrm{N_2O}}", "N₂O factor per unit of activity", "kgN₂O/t", "factors"),
    SymbolDef("GWP₁₀₀", "GWP_{100}", "100-year global warming potential of a gas", "kgCO₂e/kg gas", "factors"),
    SymbolDef("EF_valid", "EF_{\\mathrm{valid}}", "validity window [τ₀, τ₁] of an emission-factor record", "—", "factors"),
    SymbolDef("EF_grid,y", "EF_{\\mathrm{grid},y}", "location-based grid-average factor, country y, year 𝒫", "kgCO₂e/kWh", "scope2"),
    SymbolDef("EF_grid,h", "EF_{\\mathrm{grid},h}", "hourly grid factor (hourly matching, footprint based)", "kgCO₂e/kWh", "scope2"),
    SymbolDef("EF_contract", "EF_{\\mathrm{contract}}", "factor of a contractual instrument (PPA, GO, REC, tariff)", "kgCO₂e/kWh", "scope2"),
    SymbolDef("EF_residual", "EF_{\\mathrm{residual}}", "residual-mix factor covering non-claimed energy", "kgCO₂e/kWh", "scope2"),
    SymbolDef("EF_EEIO", "EF_{\\mathrm{EEIO}}", "environmentally-extended input-output factor per unit spend", "kgCO₂e/M\$", "scope3"),
    # ── emissions ────────────────────────────────────────────────────────────
    SymbolDef("Eᵢ", "E_i", "emissions of source i,  Eᵢ = Q̇ᵢ · EFᵢ", "tCO₂e/𝒫", "scope1"),
    SymbolDef("E_Scope1", "E_{\\mathrm{scope1}}", "direct emissions,  Σᵢ(Q̇ᵢ·EFᵢ) over owned/controlled sources", "tCO₂e/𝒫", "scope1"),
    SymbolDef("E_stat", "E_{\\mathrm{stat}}", "stationary combustion emissions", "tCO₂e/𝒫", "scope1"),
    SymbolDef("E_mob", "E_{\\mathrm{mob}}", "mobile combustion emissions (fleet, off-road)", "tCO₂e/𝒫", "scope1"),
    SymbolDef("E_fug", "E_{\\mathrm{fug}}", "fugitive emissions (leaks, refrigerant loss, venting)", "tCO₂e/𝒫", "scope1"),
    SymbolDef("E_flare", "E_{\\mathrm{flare}}", "flaring emissions net of DRE and combustion efficiency", "tCO₂e/𝒫", "scope1"),
    SymbolDef("E_proc", "E_{\\mathrm{proc}}", "process emissions (chemical/physical transformation)", "tCO₂e/𝒫", "scope1"),
    SymbolDef("C_elec", "C_{\\mathrm{elec}}", "purchased electricity consumption", "MWh/𝒫", "scope2"),
    SymbolDef("E_loc", "E_{\\mathrm{loc}}", "Scope 2 location-based,  Σᵢ(C_elecᵢ · EF_grid)", "tCO₂e/𝒫", "scope2"),
    SymbolDef("E_mkt", "E_{\\mathrm{mkt}}", "Scope 2 market-based,  Σᵢ(Cᵢ·EF_contract) + EF_residual·C_unclaimed", "tCO₂e/𝒫", "scope2"),
    SymbolDef("λ_cov", "\\lambda_{\\mathrm{cov}}", "share of purchased energy covered by contractual instruments", "—", "scope2"),
    SymbolDef("E_Scope3", "E_{\\mathrm{scope3}}", "value-chain emissions,  Σ over categories 1…15", "tCO₂e/𝒫", "scope3"),
    SymbolDef("cat", "\\mathrm{cat}", "Scope 3 category index, cat ∈ {1,…,15}", "—", "scope3"),
    SymbolDef("E_1⃗", "E_{\\mathrm{cat}}", "vector of the 15 category results (E_1 … E_15)", "tCO₂e/𝒫", "scope3"),
    SymbolDef("s_spend", "s_{\\mathrm{spend}}", "procurement spend, deflated to the EEIO base year", "M\$", "scope3"),
    SymbolDef("tkm", "t\\!\\cdot\\!km", "transport activity, mass × distance", "t·km", "scope3"),
    SymbolDef("pkm", "p\\!\\cdot\\!km", "passenger transport activity", "p·km", "scope3"),
    SymbolDef("DQ", "DQ", "data-quality class of an activity (1 supplier specific … 4 proxy)", "—", "scope3"),
    # ── uncertainty (GUM + Monte Carlo) ──────────────────────────────────────
    SymbolDef("σᵢ", "\\sigma_i", "standard uncertainty of input i", "unit of i", "uncertainty"),
    SymbolDef("cᵢ", "c_i", "sensitivity coefficient,  cᵢ = ∂E/∂xᵢ", "tCO₂e/unit of xᵢ", "uncertainty"),
    SymbolDef("u_c", "u_c", "combined standard uncertainty,  u_c = √(Σᵢ(cᵢσᵢ)²)", "tCO₂e", "uncertainty"),
    SymbolDef("U₉₅", "U_{95}", "expanded uncertainty at 95 %,  U₉₅ = k_cov · u_c", "tCO₂e", "uncertainty"),
    SymbolDef("k_cov", "k_{\\mathrm{cov}}", "coverage factor (2 for ~95 % normal coverage)", "—", "uncertainty"),
    SymbolDef("CV", "CV", "coefficient of variation,  CV = σ/μ", "—", "uncertainty"),
    SymbolDef("n_MC", "n_{\\mathrm{MC}}", "number of Monte-Carlo trials", "—", "uncertainty"),
    SymbolDef("ρᵢⱼ", "\\rho_{ij}", "correlation between inputs i and j (GUM covariance term)", "—", "uncertainty"),
    # ── neural soft sensor (annual analyzer campaign) ────────────────────────
    SymbolDef("x⃗", "\\vec{x}", "feature vector of continuous DCS/SCADA signals", "mixed", "softsensor"),
    SymbolDef("y", "y", "target variable, measured only during the analyzer campaign", "mixed", "softsensor"),
    SymbolDef("ŷ", "\\hat{y}", "soft-sensor prediction,  ŷ = f_θ⃗(x⃗)", "mixed", "softsensor"),
    SymbolDef("θ⃗", "\\vec{\\theta}", "all trainable parameters {W⁽ˡ⁾, b⁽ˡ⁾}", "—", "softsensor"),
    SymbolDef("W⁽ˡ⁾", "W^{(l)}", "weight matrix of layer l,  W⁽ˡ⁾ ∈ ℝ^{n_l × n_{l−1}}", "—", "softsensor"),
    SymbolDef("b⁽ˡ⁾", "b^{(l)}", "bias vector of layer l", "—", "softsensor"),
    SymbolDef("z⁽ˡ⁾", "z^{(l)}", "pre-activation,  z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾", "—", "softsensor"),
    SymbolDef("a⁽ˡ⁾", "a^{(l)}", "activation,  a⁽ˡ⁾ = g(z⁽ˡ⁾)", "—", "softsensor"),
    SymbolDef("δ⁽ˡ⁾", "\\delta^{(l)}", "error signal,  δ⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾  (back-propagation)", "—", "softsensor"),
    SymbolDef("η", "\\eta", "learning rate of the optimiser", "—", "softsensor"),
    SymbolDef("β₁, β₂", "\\beta_1,\\beta_2", "Adam decay rates for the 1st / 2nd moment", "—", "softsensor"),
    SymbolDef("m̂, v̂", "\\hat{m},\\hat{v}", "bias-corrected Adam moment estimates", "—", "softsensor"),
    SymbolDef("λ_L2", "\\lambda_{L2}", "L2 weight-decay coefficient (ridge penalty Σ‖W‖²)", "—", "softsensor"),
    SymbolDef("L", "\\mathcal{L}", "loss, e.g. Gaussian negative log-likelihood", "—", "softsensor"),
    SymbolDef("σ̂", "\\hat{\\sigma}", "estimated residual standard deviation (aleatoric noise)", "mixed", "softsensor"),
    SymbolDef("N_camp", "N_{\\mathrm{camp}}", "campaign samples, 7 d · 24 h = 168 per source", "—", "softsensor"),
    SymbolDef("D²_M", "D^2_M", "Mahalanobis distance of a sample from the campaign cloud", "—", "softsensor"),
    SymbolDef("ε", "\\varepsilon", "numerical floor (division by zero / log 0 protection)", "—", "softsensor"),
    # ── anomaly detection ────────────────────────────────────────────────────
    SymbolDef("r̂ₜ", "\\hat{r}_t", "residual,  r̂ₜ = yₜ − ŷₜ", "mixed", "anomaly"),
    SymbolDef("zₜ", "z_t", "EWMA statistic,  zₜ = λ·r̂ₜ + (1−λ)·z_{t−1}", "—", "anomaly"),
    SymbolDef("Sₜ", "S_t", "CUSUM statistic (two-sided with reference value k)", "—", "anomaly"),
    SymbolDef("T²", "T^2", "Hotelling statistic of a multivariate observation", "—", "anomaly"),
    SymbolDef("h_CL", "h_{\\mathrm{CL}}", "control limit calibrated to a target false-alarm rate", "—", "anomaly"),
    SymbolDef("ARL₀", "ARL_0", "average run length between false alarms", "samples", "anomaly"),
    SymbolDef("SPE", "SPE", "squared prediction error (Q-statistic) of PCA / autoencoder", "—", "anomaly"),
    # ── audit trail, security, data quality ──────────────────────────────────
    SymbolDef("hₙ", "h_n", "hash of audit entry n,  hₙ = H(hₙ₋₁ ‖ entryₙ)", "—", "audit"),
    SymbolDef("M_merkle", "M_{\\mathrm{merkle}}", "Merkle root over a batch of audit entries", "—", "audit"),
    SymbolDef("κ_hmac", "\\kappa_{\\mathrm{hmac}}", "HMAC-SHA256 key signing evidence manifests", "—", "audit"),
    SymbolDef("τ_ts", "\\tau_{\\mathrm{ts}}", "trusted timestamp of an audit entry", "UTC", "audit"),
    SymbolDef("a_actor", "a", "actor (user or service) performing an action", "—", "audit"),
    SymbolDef("π_role", "\\pi", "role: viewer, analyst, approver, auditor, admin", "—", "audit"),
    SymbolDef("rᵢⱼ", "r_{ij}", "permission matrix: role i may perform action j", "—", "audit"),
    SymbolDef("salt", "salt", "per-user random salt for PBKDF2 password derivation", "—", "audit"),
    SymbolDef("dᵢⱼ", "d_{ij}", "data-quality score of activity i on criterion j", "—", "audit"),
    SymbolDef("wⱼ", "w_j", "weight of data-quality criterion j", "—", "audit"),
    SymbolDef("DQI", "DQI", "weighted data-quality index,  DQIᵢ = Σⱼ wⱼ dᵢⱼ / Σⱼ wⱼ", "—", "audit"),
]

# ── legend API ───────────────────────────────────────────────────────────────
"Areas of the engine, in presentation order."
const NOTATION_AREAS = ["boundary", "activity", "factors", "scope1", "scope2", "scope3",
                        "uncertainty", "softsensor", "anomaly", "audit"]

"""
    notation_table(; area=nothing) -> Vector{Vector{String}}

The legend as a table with the columns `Symbol, LaTeX, Meaning, Unit, Area`,
optionally filtered to one `area`.
"""
function notation_table(; area::Union{Nothing,AbstractString}=nothing)
    rows = [["Symbol", "LaTeX", "Meaning", "Unit", "Area"]]
    for s in NOTATION
        (area === nothing || s.area == area) || continue
        push!(rows, [s.symbol, s.latex, s.meaning, s.unit, s.area])
    end
    rows
end

"""
    notation_markdown(; area=nothing) -> String

Markdown rendering of the legend — used by notebook 00 and by the PDF build so
that the printed report documents its own symbols.
"""
function notation_markdown(; area::Union{Nothing,AbstractString}=nothing)
    rows = notation_table(area=area)
    io = IOBuffer()
    println(io, "| ", join(rows[1], " | "), " |")
    println(io, "|", join(fill("---", length(rows[1])), "|"), "|")
    for r in rows[2:end]
        println(io, "| `", r[1], "` | `", r[2], "` | ", r[3], " | ", r[4], " | ", r[5], " |")
    end
    String(take!(io))
end

"""
    notation_latex(; area=nothing) -> String

LaTeX longtable of the legend, for a printed appendix.
"""
function notation_latex(; area::Union{Nothing,AbstractString}=nothing)
    rows = notation_table(area=area)
    io = IOBuffer()
    println(io, "\\begin{longtable}{llll}")
    println(io, "\\textbf{Symbol} & \\textbf{Meaning} & \\textbf{Unit} & \\textbf{Area} \\\\")
    for r in rows[2:end]
        println(io, "\$", r[2], "\$ & ", replace(r[3], "_" => "\\_"), " & ", r[4], " & ", r[5], " \\\\")
    end
    println(io, "\\end{longtable}")
    String(take!(io))
end

"""
    symbol_list(area) -> Vector{String}

All symbols of one area (used by the notebooks to show their own vocabulary).
"""
symbol_list(area::AbstractString) = [s.symbol for s in NOTATION if s.area == area]

"""
    check_notation() -> Vector{String}

Self-test of the legend: every `symbol` field must be a legal Julia identifier,
so that the legend and the source can never drift apart. Returns the list of
offending entries (empty when the legend is consistent).
"""
function check_notation()
    bad = String[]
    for s in NOTATION
        # a symbol field may contain a spaced form like "E_1 ⃗" or "β₁, β₂":
        # every whitespace-separated token must parse as an identifier
        for tok in split(s.symbol, r"[\s,]+")
            isempty(tok) && continue
            if !(try; Meta.parse(tok * " = 1.0"); true; catch; false; end)
                push!(bad, s.symbol)
            end
        end
    end
    unique(bad)
end

