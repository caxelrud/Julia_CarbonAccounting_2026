# ══════════════════════════════════════════════════════════════════════════════
#  make_data.jl — synthetic but realistic source data for the whole platform
#
#  Run:  julia --project=. scripts/make_data.jl
#
#  Everything is generated from a fixed seed, so re-running the script reproduces
#  the repository's data exactly. The files are deliberately *messy* the way real
#  exports are: semicolon delimiters, decimal commas, units glued to values,
#  multi-language headers, a few rows that must be quarantined, mixed units
#  between files, and hourly data with injected anomalies.
# ══════════════════════════════════════════════════════════════════════════════
using Random, Dates, Statistics, Printf, DataFrames, CSV

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RAW = joinpath(ROOT, "data", "raw")
const REF = joinpath(ROOT, "data", "reference")
mkpath(RAW); mkpath(REF)

rng = MersenneTwister(2026)
const YEAR = 2024

# ── master data: the portfolio ───────────────────────────────────────────────
# entity, site, country, equity share, consent (months), production t, revenue MUSD, FTE
const SITES = [
    ("AcmeIndustrial_SA", "SAO-01", "BRA", 1.00, 2_450_000.0, 780.0, 640),
    ("AcmeIndustrial_SA", "SAO-02", "BRA", 1.00, 1_180_000.0, 390.0, 310),
    ("AcmeEurope_GmbH", "HAM-01", "DEU", 1.00, 980_000.0, 620.0, 380),
    ("AcmeEurope_GmbH", "ROT-02", "NLD", 0.60, 540_000.0, 300.0, 150),
    ("AcmeAfrica_Pty", "JNB-01", "ZAF", 0.74, 760_000.0, 210.0, 420),
    ("AcmeAsia_Co", "SHA-03", "CHN", 0.45, 620_000.0, 180.0, 260),
]
master = DataFrame(entity=[s[1] for s in SITES], site=[s[2] for s in SITES],
    country=[s[3] for s in SITES], equity_share=[s[4] for s in SITES],
    production_t=[s[5] for s in SITES], revenue_usd=[s[6] * 1e6 for s in SITES],
    employees=[s[7] for s in SITES])
CSV.write(joinpath(REF, "master_data.csv"), master)

# ── reference: price index for spend-based methods ───────────────────────────
cpi = DataFrame(year=[2020, 2021, 2022, 2023, 2024, 2025, 2026],
    cpi_usd=[100.0, 104.7, 113.2, 117.9, 121.9, 125.0, 128.2],
    fx_eur_usd=[1.142, 1.183, 1.053, 1.082, 1.082, 1.090, 1.100],
    fx_brl_usd=[0.193, 0.184, 0.192, 0.202, 0.196, 0.198, 0.200])
CSV.write(joinpath(REF, "price_index.csv"), cpi)

println("master data + price index written")

# ── Scope 1: fuel and process activity, German/Portuguese headers, ';' ───────
# month → seasonal fuel factor (winter heating, plant load)
seasonal(m) = 1.0 + 0.28 * cos(2π * (m - 1) / 12)

fuel_rows = Any[]
for (entity, site, country, ©, _, _, _) in SITES
    scale = 1.0 + 0.4 * rand(rng)
    for m in 1:12
        # natural gas, energy in GJ, comma decimal, period label 01.03.2024-31.03.2024
        gas = 18_000 * scale * seasonal(m) * (0.9 + 0.2 * rand(rng))
        push!(fuel_rows, ("Erdgas", site, "Erdgas Kessel 1", @sprintf("%.0f", gas), "GJ",
            "01." * lpad(string(m), 2, "0") * ".$YEAR-" * lpad(string(daysinmonth(YEAR, m)), 2, "0") *
            "." * lpad(string(m), 2, "0") * ".$YEAR", "Wintershall", country, 2, "Kesselhaus"))
        # diesel for the fleet, litres
        diesel = 4_200 * scale * (0.85 + 0.3 * rand(rng))
        push!(fuel_rows, ("Frota", site, "Diesel Gerador", @sprintf("%.0f", diesel), "L",
            string(YEAR, "-", lpad(string(m), 2, "0")), "Petrobras", country, 2, "Frota"))
    end
    # process emissions and fugitives: free-text source types resolved by alias
    push!(fuel_rows, ("Clinker Ofen", site, "Clinker", @sprintf("%.0f", 42_000 * scale), "t",
        string(YEAR), "intern", country, 2, "Prozess"))
    push!(fuel_rows, ("Salpetersäure", site, "HNO3 Anlage", @sprintf("%.0f", 3_100 * scale), "t",
        string(YEAR), "intern", country, 2, "Prozess N2O"))
    push!(fuel_rows, ("Refrigerante", site, "R410A Nachfüllung", @sprintf("%.1f", 120 * scale),
        "kg", string(YEAR), "ClimaTech", country, 2, "Kaeltemaschine"))
    push!(fuel_rows, ("Flare", site, "Fackelgas", @sprintf("%.0f", 2.4e6 * scale), "m3(n)",
        string(YEAR), "intern", country, 2, "Fackel"))
    push!(fuel_rows, ("LDAR", site, "Ventil Survey Q1", @sprintf("%.0f", 1_850 * scale),
        "component", string(YEAR, "-Q1"), "intern", country, 4, "LDAR Kampagne"))
end
# deliberately broken rows for the quarantine demo
push!(fuel_rows, ("Wundermittel", "HAM-01", "Wundermittel Kessel", "12.500", "GJ", "$YEAR-05", "Unbekannt", "DEU", 3, "Test"))
push!(fuel_rows, ("Erdgas", "HAM-01", "Erdgas Kessel 1", "-8.000", "GJ", "$YEAR-06", "Wintershall", "DEU", 2, "Negativ"))
push!(fuel_rows, ("Erdgas", "HAM-01", "Erdgas Kessel 1", "9.100", "Epstein-Barrel", "$YEAR-06", "Wintershall", "DEU", 2, "Unbekannte Einheit"))

open(joinpath(RAW, "activity_erp_2024.csv"), "w") do io
    println(io, "Kategorie;Werk;Kraftstoff;Verbrauch;Einheit;Zeitraum;Lieferant;Land;Datenqualität;Anmerkung")
    for r in fuel_rows
        println(io, join(r, ";"))
    end
end

# ── Scope 2: monthly utility data, mixed units, contractual instruments ──────
open(joinpath(RAW, "electricity_utility_2024.csv"), "w") do io
    println(io, "Tipo,Site,Country,Month,Consumption,Unit,Instrument,Supplier")
    for (entity, site, country, ©, _, _, _) in SITES
        base_mwh = country == "BRA" ? 2_400.0 : country == "DEU" ? 1_450.0 :
                   country == "ZAF" ? 3_100.0 : country == "CHN" ? 2_800.0 : 1_700.0
        for m in 1:12
            c = base_mwh * seasonal(m) * (0.9 + 0.2 * rand(rng))
            unit = rand(rng) < 0.5 ? "kWh" : "MWh"
            val = unit == "kWh" ? c * 1000 : c
            instrument = if site in ("SAO-01", "HAM-01") && m >= 4
                rand(rng) < 0.8 ? "ppa_renewable" : "green_tariff"
            else
                ""
            end
            supplier = site in ("SAO-01", "HAM-01") ? "RenewCo PPA" : "Utility $(country)"
            println(io, join(["electricity", site, country, @sprintf("%04d-%02d", YEAR, m),
                              @sprintf("%.0f", val), unit, instrument, supplier], ","))
        end
    end
end
println("electricity_utility_2024.csv written")

# ── Scope 3: procurement spend (EEIO), multiple currencies and years ─────────
proc_catalogue = [("stainless steel plate", "steel", 620.0), ("cement bulk", "cement", 480.0),
    ("industrial chemicals", "chemicals", 350.0), ("polymer resin", "plastics", 410.0),
    ("electronic control unit", "electronics", 260.0), ("pump overhaul", "machinery", 180.0),
    ("facility management", "services", 96.0), ("IT consulting", "services", 140.0),
    ("corrugated packaging", "paper", 72.0), ("cold-chain logistics", "transport_services", 210.0),
    ("mining explosives", "mining", 130.0), ("pharmaceutical supplies", "pharmaceuticals", 88.0),
    ("agricultural fertiliser", "agriculture", 160.0), ("cotton fabric", "textiles", 120.0)]
open(joinpath(RAW, "procurement_2024.csv"), "w") do io
    println(io, "Tipo,PO_id,Site,Fornecedor,Description,Amount,Currency,Year")
    for i in 1:42
        (desc, sector, base) = proc_catalogue[rand(rng, 1:length(proc_catalogue))]
        (entity, site, country, ©, _, _, _) = SITES[rand(rng, 1:length(SITES))]
        cur = country == "BRA" ? "BRL" : country == "DEU" || country == "NLD" ? "EUR" :
              country == "ZAF" ? "ZAR" : country == "CHN" ? "CNY" : "USD"
        amount = base * 1000 * (0.4 + 1.4 * rand(rng))
        yr = rand(rng) < 0.75 ? YEAR : YEAR - 1
        println(io, join(["purchased_goods_spend", @sprintf("PO-%05d", 1000 + i), site,
                          "Supplier $(100 + i)", desc, @sprintf("%.0f", amount), cur,
                          string(yr)], ","))
    end
end
println("procurement_2024.csv written")

# ── Scope 3: transport, travel, commuting ───────────────────────────────────
open(joinpath(RAW, "logistics_travel_2024.csv"), "w") do io
    println(io, "Tipo,Site,Item,Quantity,Unit,Period")
    for (entity, site, country, ©, _, _, _) in SITES
        # inbound freight (cat 4) and outbound distribution (cat 9): t·km by mode
        for (mode, tkm) in [("hgv_diesel", 8.5e5), ("sea_container", 4.0e6), ("rail_freight", 2.2e6)]
            println(io, join(["upstream_transport", site, mode,
                              @sprintf("%.0f", tkm * (0.7 + 0.6 * rand(rng))), "tkm", string(YEAR)], ","))
        end
        for (mode, tkm) in [("hgv_diesel", 6.0e5), ("sea_container", 2.5e6)]
            println(io, join(["downstream_transport", site, mode,
                              @sprintf("%.0f", tkm * (0.8 + 0.4 * rand(rng))), "tkm", string(YEAR)], ","))
        end
        # business travel (cat 6) and commuting (cat 7): p·km
        println(io, join(["business_travel_air", site, "air_short_haul",
                          @sprintf("%.0f", 85_000 * (0.6 + 0.8 * rand(rng))), "pkm", string(YEAR)], ","))
        println(io, join(["business_travel_air", site, "air_long_haul",
                          @sprintf("%.0f", 420_000 * (0.5 + rand(rng))), "pkm", string(YEAR)], ","))
        println(io, join(["business_travel_land", site, "car_average",
                          @sprintf("%.0f", 260_000 * (0.6 + 0.8 * rand(rng))), "pkm", string(YEAR)], ","))
        for (mode, pkm) in [("car_average", 3.2e6), ("bus", 9.0e5), ("rail", 1.4e6)]
            println(io, join(["employee_commuting", site, mode,
                              @sprintf("%.0f", pkm * (0.7 + 0.6 * rand(rng))), "pkm", string(YEAR)], ","))
        end
    end
end
println("logistics_travel_2024.csv written")

# ── Scope 3: waste (cat 5) ──────────────────────────────────────────────────
open(joinpath(RAW, "waste_2024.csv"), "w") do io
    println(io, "Tipo,Site,Route,Mass,Unit,Period")
    for (entity, site, country, ©, _, _, _) in SITES
        for (route, tons) in [("landfill_msw", 620.0), ("recycling", 410.0), ("incineration", 180.0)]
            println(io, join(["waste", site, route, @sprintf("%.0f", tons * (0.6 + 0.9 * rand(rng))),
                              "t", string(YEAR)], ","))
        end
    end
end
println("waste_2024.csv written")

# ── the annual analyzer campaign: one week, hourly, with the measured variable ─
#  Physics of the campaign signal: the methane mole fraction in the flare feed
#  depends on combustion oxygen, stack temperature, load and gas flow —
#      y = 1.4 − 0.010·O₂ + 0.0009·(T − 320) + 0.45·cos(load/22) + 2.5e-5·V̇ + ε
#  The continuous tags are the soft-sensor inputs; y is what the analyzer sees.
function ch4_fraction(load, T, o2, flow)
    1.4 - 0.010 * o2 + 0.0009 * (T - 320.0) + 0.45 * cos(load / 22.0) + 2.5e-5 * flow
end
campaign_hours = collect(1:168)
camp_tags = zeros(length(campaign_hours), 6)
camp_y = zeros(length(campaign_hours))
for (i, h) in enumerate(campaign_hours)
    load = 55.0 + 22.0 * sin(2π * h / 24) + 4.0 * randn(rng)      # 35 … 85 %
    T = 300.0 + 0.55 * load + 3.0 * randn(rng)
    o2 = 4.6 - 0.020 * load + 0.10 * randn(rng)
    flow = 2_200.0 + 38.0 * load + 180.0 * randn(rng)
    press = 2.4 + 0.004 * flow / 100 + 0.05 * randn(rng)
    feed = 8.5 + 0.10 * load + 0.3 * randn(rng)
    camp_tags[i, :] = [load, T, o2, flow, press, feed]
    camp_y[i] = ch4_fraction(load, T, o2, flow) + 0.035 * randn(rng)
end
CSV.write(joinpath(RAW, "analyzer_campaign_2024.csv"),
    DataFrame(hour=campaign_hours,
        timestamp=[string(DateTime(YEAR, 3, 4) + Hour(h - 1)) for h in campaign_hours],
        load_pct=camp_tags[:, 1], stack_temp_C=camp_tags[:, 2], o2_pct=camp_tags[:, 3],
        flare_flow_m3h=camp_tags[:, 4], pressure_bar=camp_tags[:, 5], feedstock_tph=camp_tags[:, 6],
        ch4_fraction_pct=camp_y))
println("analyzer_campaign_2024.csv: ", length(campaign_hours), " campaign hours")

# ── continuous DCS data for the whole year, with injected anomalies ──────────
hours = 1:(365 * 24)
X = zeros(length(hours), 6)
for t in hours
    day, hod = div(t - 1, 24) + 1, mod(t - 1, 24)
    season = 1.0 + 0.18 * cos(2π * (day - 15) / 365)
    diurnal = 1.0 + 0.22 * sin(2π * (hod - 6) / 24)
    load = clamp(78.0 * season * diurnal + 4.0 * randn(rng), 25.0, 104.0)
    T = 300.0 + 0.55 * load + 4.0 * randn(rng)
    o2 = 4.6 - 0.020 * load + 0.12 * randn(rng)
    flow = 2_200.0 + 38.0 * load + 240.0 * randn(rng)
    press = 2.4 + 0.004 * flow / 100 + 0.06 * randn(rng)
    feed = 8.5 + 0.10 * load + 0.4 * randn(rng)
    X[t, :] = [load, T, o2, flow, press, feed]
end
# four injected episodes: sensor drift, correlation break, leak, thermal spike
anomaly_windows = [40 * 24 .+ (1:30), 120 * 24 .+ (1:20), 210 * 24 .+ (1:24), 300 * 24 .+ (1:12)]
for (i, w) in enumerate(anomaly_windows)
    for t in w
        i == 1 && (X[t, 3] += 1.6)          # O₂ sensor reads high: drift
        i == 2 && (X[t, 4] -= 900.0)        # flow falls while load rises: correlation break
        i == 3 && (X[t, 5] -= 0.45)         # pressure falls: leak
        i == 4 && (X[t, 2] += 26.0)         # short thermal spike
    end
end
CSV.write(joinpath(RAW, "dcs_hourly_2024.csv"),
    DataFrame(timestamp=[string(DateTime(YEAR, 1, 1) + Hour(t - 1)) for t in hours],
        load_pct=X[:, 1], stack_temp_C=X[:, 2], o2_pct=X[:, 3],
        flare_flow_m3h=X[:, 4], pressure_bar=X[:, 5], feedstock_tph=X[:, 6]))
println("dcs_hourly_2024.csv: ", length(hours), " hours, 4 injected anomaly episodes")

# ── the emission-factor library, dumped so the factor set is inspectable ─────
using CarbonAccounting
CA = CarbonAccounting
CSV.write(joinpath(REF, "emission_factors.csv"), CA.factors_dataframe(CA.default_library()))
println("emission_factors.csv: ", length(CA.default_library()), " factor records (",
        CA.default_library().version, ")")
println("\nALL DATA WRITTEN to:")
println("  ", RAW)
println("  ", REF)
