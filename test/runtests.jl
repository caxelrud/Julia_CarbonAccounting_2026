# ══════════════════════════════════════════════════════════════════════════════
#  runtests.jl — the test suite
#
#  Run:  julia --project=. test/runtests.jl
#        julia --project=. -e 'using Pkg; Pkg.test()'
#
#  The tests check the *properties that matter* rather than fixed numbers: that the
#  dimensional algebra rejects wrong pairings, that the hash chain detects tampering,
#  that GUM and Monte-Carlo uncertainties agree, that the soft sensor beats a constant
#  predictor and states an honest interval, and that the detectors find anomalies they
#  were not told about.
# ══════════════════════════════════════════════════════════════════════════════
using CarbonAccounting, Test, Dates, Statistics, Random, DataFrames, CSV

const RAW = CarbonAccounting.datadir("raw")
const FILES = ["activity_erp_2024.csv", "electricity_utility_2024.csv", "procurement_2024.csv",
               "logistics_travel_2024.csv", "waste_2024.csv"]

"The shared fixture: the ingested records of the synthetic portfolio."
function test_records()
    all(isfile(joinpath(RAW, f)) for f in FILES) ||
        error("data/raw is empty — run `julia --project=. scripts/make_data.jl` first")
    ingest_files([joinpath(RAW, f) for f in FILES]; entity_default="AcmeIndustrial_SA")
end

@testset "CarbonAccounting.jl" begin

    @testset "notation" begin
        @test isempty(check_notation())
        @test length(NOTATION) > 80
        @test Set(NOTATION_AREAS) ⊆ Set(s.area for s in NOTATION)
        @test notation_table()[1] == ["Symbol", "LaTeX", "Meaning", "Unit", "Area"]
        @test occursin("scope1", notation_markdown(area="scope1"))
        @test Σᵢ(Float64[]) == 0.0                      # an empty sum is zero, not an error
        @test Σᵢ(abs, [-1.0, 2.0]) == 3.0
        @test GWP_AR6.CH₄_fossil < GWP_AR5.CH₄_fossil   # the AR6 set is the strictest
        @test GWP_AR4.CH₄_fossil < GWP_AR5.CH₄_fossil
        @test co2e(; CH₄=1.0) ≈ 29.8
        @test co2e(; CH₄=1.0, fossil=false) ≈ 27.0
        @test co2e(; CH₄=1.0, gwp=GWP_AR4) ≈ 25.0
        @test co2e(; CO₂=1.0, N₂O=1.0) ≈ 274.0
    end

    @testset "units: the E = Q̇·EF algebra" begin
        @test val_canon(parse_quantity("1,250 MWh")) ≈ 4_500.0
        @test val_canon(parse_quantity("2,204.6 lb")) ≈ 0.99995 atol=1e-4
        @test val_canon(parse_quantity("1 scf")) ≈ 0.0283168 atol=1e-6
        @test value_in(Quantity(0.0, "°C"), "K") ≈ 273.15
        @test value_in(Quantity(32.0, "°F"), "°C") ≈ 0.0 atol=1e-6
        @test has_unit("tCO2e") && has_unit("Nm3") && !has_unit("Epstein-Barrel")
        @test emission(q"1250 MWh", q"0.42 kgCO₂e/kWh").val ≈ 525.0
        @test emission(q"4500 GJ", q"0.42 kgCO₂e/GJ").val ≈ 1.89
        # a factor expressed in another unit *of the same dimension* is converted…
        @test emission(q"1250 MWh", q"0.42 kgCO₂e/GJ").val ≈ 1.89
        # …while a factor of the wrong dimension is refused
        @test_throws DimensionMismatch emission(q"1250 MWh", q"0.42 kgCO₂e/t")
        @test_throws DimensionMismatch emission(q"1000 t", q"0.42 kgCO₂e/kWh")
        @test emission_gas(q"100 t", q"2.0 kgCH₄/t").val ≈ 5.96
        @test emission(q"4.0e6 t·km", q"0.107 kgCO₂e/t·km").val ≈ 428.0
        # quantity arithmetic keeps dimensions
        @test (q"1 MWh" + q"1000 kWh") ≈ q"2 MWh"
        @test_throws DimensionMismatch q"1 MWh" + q"1 t"
        @test canonical(q"2500 kg").val ≈ 2.5
        @test parse_quantity("1.234,5 GJ").val ≈ 1234.5
        @test parse_compound("kgCO2e/kWh").gas == "CO2e"
        @test parse_compound("kgCH4/t").gas == "CH4"
        @test adjust_ef(0.4, parse_compound("kgCO₂e/kWh"), "MWh") ≈ 400.0
    end

    @testset "security and the audit trail" begin
        analyst = Actor("u001", "A. Analyst", :analyst)
        approver = Actor("u002", "B. Approver", :approver)
        @test can(analyst, CalculateAction)
        @test !can(analyst, ApproveAction)
        @test_throws AccessDenied authorize(analyst, ApproveAction; context="t")
        @test authorize(approver, ApproveAction; context="t")
        @test_throws ArgumentError check_segregation_of_duties(analyst, analyst)
        @test check_segregation_of_duties(analyst, approver)
        s = open_session(analyst; ttl_minutes=1, at=DateTime(2024, 1, 1, 12, 0))
        @test valid(s; at=DateTime(2024, 1, 1, 12, 0, 30))
        @test !valid(s; at=DateTime(2024, 1, 1, 12, 2))
        cred = hash_password("u001", "s3cret"; iterations=1_000)
        @test check_password(cred, "s3cret")
        @test !check_password(cred, "s3cre")
        @test length(cred.hash) == 32 && cred.iterations == 1_000
        @test redact("token=abc123 me@example.com") == "token=<redacted> <email>"

        l = AuditLedger(hmac_key=Vector{UInt8}(codeunits("test-key")))
        for i in 1:5
            record!(l, analyst, "calculate", "scope1"; why="run $i", after="E = $i")
        end
        @test length(l) == 5
        @test first(verify_chain(l))
        @test l.entries[2].prev_hash == l.entries[1].hash
        @test head_hash(l) == l.entries[end].hash
        m = seal(l)
        @test verify_seal(m, l)
        @test m.merkle == merkle_root(l)
        # the analyst cannot forge an admin-level entry
        @test_throws AccessDenied record!(l, analyst, "adjust", "x"; why="y")
        tamper!(l, 3; after="1 tCO₂e (forged)")
        ok, broken = verify_chain(l)
        @test !ok && broken == 3
        # two independent layers: editing content breaks the *chain* (the stored hashes
        # no longer match their content), while the seal still holds — a forger would
        # have to recompute every hash, and that breaks the seal instead.
        @test verify_seal(m, l)
        reforged = AuditLedger(hmac_key=l.hmac_key)
        for e in l.entries
            record!(reforged, Actor(e.actor_id, e.actor_name, Symbol(e.role)), e.action,
                    e.entity; why=e.why, before=e.before, after=e.after,
                    source_hash=e.source_hash, ts=e.ts)
        end
        @test first(verify_chain(reforged))     # the re-forged chain is self-consistent…
        @test !verify_seal(m, reforged)         # …but it no longer matches the sealed manifest
        path = joinpath(CarbonAccounting.builddir("test"), "ledger.json")
        save_ledger(l, path)
        reloaded = load_ledger(path; hmac_key=l.hmac_key)
        @test length(reloaded) == 5
        @test nrow(audit_trail_table(reloaded)) == 5
        @test can(Actor("a", "Admin", :admin), AdminAction)
    end

    @testset "ingest: parsing, normalisation, quarantine" begin
        @test parse_number("1,234.5") ≈ 1234.5
        @test parse_number("1.234,5") ≈ 1234.5
        @test parse_number("(1 250,75)") ≈ -1250.75
        @test parse_number("12,5%") ≈ 12.5
        @test parse_date("31.03.2024") == Date(2024, 3, 31)
        @test parse_date("2024-03-31") == Date(2024, 3, 31)
        @test parse_date("31/03/2024"; dayfirst=false) == Date(2024, 3, 31)
        @test_throws ArgumentError parse_date("01")
        @test parse_period("2024-Q1") == (Date(2024, 1, 1), Date(2024, 3, 31))
        @test parse_period("2024-03") == (Date(2024, 3, 1), Date(2024, 3, 31))
        @test parse_period("2024") == (Date(2024, 1, 1), Date(2024, 12, 31))
        @test parse_period("01.03.2024-31.03.2024") == (Date(2024, 3, 1), Date(2024, 3, 31))
        @test header_key("Kraftstoff") == "kraftstoff"
        @test lookup_source("Erdgas Kessel 1").key == "natural_gas"
        @test lookup_source("Frota").key == "diesel_fleet"
        @test lookup_source("Clinker Ofen").key == "process_cement"
        @test lookup_source("Wundermittel") === nothing
        @test iso_country("BR") == "BRA"
        @test iso_country("deutschland") == "DEU"

        rep = test_records()
        @test length(rep.records) > 200
        @test length(rep.files) == 5
        @test any(q -> occursin("Wundermittel", q.reason), rep.quarantined)
        @test any(q -> occursin("Epstein-Barrel", q.reason), rep.quarantined)
        r = first(rep.records)
        @test r.Q̇ᵢ isa Quantity && !isempty(r.unit_canon)
        @test !isempty(r.source_hash) && r.lineno > 1
        @test all(1 <= x.DQ <= 4 for x in rep.records)
        @test all(x.scope in (:scope1, :scope2, :scope3) for x in rep.records)
        @test all(!isempty(x.source_hash) for x in rep.records)
        issues = validate_records(rep.records; year=2024)
        @test any(i -> i.severity == :error, issues)
        @test nrow(records_dataframe(rep.records)) == length(rep.records)
        uniq, dropped = deduplicate(vcat(rep.records, rep.records[1:3]))
        @test length(uniq) == length(rep.records) && length(dropped) == 3
    end

    @testset "factors: library, precedence, dimensional fit" begin
        lib = default_library()
        @test length(lib) > 200
        @test lib.version == "2026.1"
        @test nrow(factors_dataframe(lib)) == length(lib)
        @test "note" in names(factors_dataframe(lib))
        f = lookup_factor(lib, "natural_gas", "natural_gas"; region="GLOBAL")[1]
        @test f.value ≈ 56.1
        @test applies_on(f, Date(2024, 6, 1))
        @test !applies_on(f, Date(2031, 1, 1))               # validity window
        @test compatibility(f, "GJ") && !compatibility(f, "MWh") == false
        @test compatibility(lookup_factor(lib, "electricity", "grid_average";
                                          region="DEU")[1], "kWh")
        # region precedence: a regional record outranks GLOBAL
        eu = lookup_factor(lib, "electricity", "grid_average"; region="DEU", date=Date(2024, 6, 1))
        @test eu[1].region == "DEU"
        @test any(x -> x.region == "GLOBAL", eu)
        # resolution: exact item, default item, EEIO sector, mobile fallback
        @test resolve_factor(lib, SOURCE_RULES["refrigerant"]; item="R410A").item == "R410A"
        @test resolve_factor(lib, SOURCE_RULES["waste"]).item == "landfill_msw"
        @test resolve_factor(lib, SOURCE_RULES["upstream_transport"]).item == "hgv_diesel"
        @test resolve_factor(lib, SOURCE_RULES["purchased_goods_spend"];
                             item="stainless steel plate").item == "steel"
        @test sector_hint("consultoria jurídica") == "services"
        @test sector_hint("polymer resin") == "plastics"
        # the market-based factor differs from the location-based one
        loc = resolve_factor(lib, SOURCE_RULES["electricity"]; region="DEU", date=Date(2024, 6, 1))
        mkt = resolve_factor(lib, SOURCE_RULES["electricity"]; region="DEU",
                             date=Date(2024, 6, 1), market=:market)
        @test mkt.value > loc.value
        @test resolve_factor(lib, SOURCE_RULES["ppa_electricity"]; item="ppa_renewable").value == 0.0
        # deflation
        idx = PriceIndex(2024, Dict(2023 => 117.9, 2024 => 121.9))
        @test deflate(1_000_000, 2023, idx) ≈ 1_033_927.0 atol=1.0
        # export/import round trip
        out = joinpath(CarbonAccounting.builddir("test"), "factors.csv")
        export_factors(lib, out)
        back = load_factors(out)
        @test length(back) == length(lib)
        @test back.records[1].value == lib.records[1].value
    end

    @testset "scope 1 engine" begin
        @test combustion_emissions(1.0; EF_CO₂=56.1).E ≈ 0.0561
        r = combustion_emissions(100.0; EF_CO₂=56.1, EF_CH₄=1e-3, EF_N₂O=1e-4, OF=0.98, η_b=0.9)
        @test r.E > r.E_CO₂                                  # CH₄/N₂O add to CO₂
        @test r.Q̇_in ≈ 100.0 / 0.9
        flare = flare_emissions(1.0e6; EF_flare=1.85, DRE=0.98, η_comb=0.995)
        @test flare.E ≈ flare.E_combusted + flare.E_slip
        @test flare.φ ≈ 0.98 * 0.995
        @test flare.E_slip > 0                               # uncombusted CH₄ is never zero
        ref = fugitive_refrigerant(; C_start=1_000, C_purchased=100, C_end=1_050, GWP₁₀₀=2088)
        @test ref.m_lost ≈ 50.0
        @test ref.E ≈ 50 * 2088 / 1000
        @test fugitive_ldar(100; leak_rate=0.5).m_CH₄ ≈ 50.0
        @test process_emissions(1_000; EF=0.525, gas="CO2", unit="tCO₂/t").E ≈ 525.0
        @test process_emissions(1_000; EF=5.7, gas="N2O", unit="kgN₂O/t").E ≈ 1_000 * 5.7 * 273 / 1_000

        scope1 = compute_scope1(test_records().records; date=Date(2024, 12, 31))
        @test scope1.E_Scope1 > 0
        @test scope1.memo_biogenic >= 0
        @test all(haskey(scope1.by_source, k) for k in keys(scope1.by_source))
        @test all(r.scope == :scope1 for r in scope1.rows)
        @test all(r.ef_value > 0 && !isempty(r.ef_id) for r in scope1.rows)
        @test all(0 <= r.u_rel < 2 for r in scope1.rows)
        @test nrow(scope1_detail(scope1)) == length(scope1.rows)
    end

    @testset "scope 2 engine: the dual report" begin
        records = test_records().records
        scope2 = compute_scope2(records; date=Date(2024, 12, 31))
        methods = Set(r.method for r in scope2.rows)
        @test "location-based" in methods && "market-based" in methods
        @test length(scope2.rows) == 2 * count(r -> r.scope == :scope2, records)
        @test scope2.E_loc > 0 && scope2.E_mkt > 0
        @test 0 <= scope2.λ_cov <= 1
        @test scope2.C_elec > 0
        @test nrow(scope2.by_site) > 0
        @test nrow(scope2_detail(scope2; which=:location)) == count(r -> r.method == "location-based",
                                                                   scope2.rows)
    end

    @testset "scope 3 engine and the inventory rollup" begin
        records = test_records().records
        scope3 = compute_scope3(records; date=Date(2024, 12, 31), hotspot_share=0.8)
        @test scope3.E_Scope3 > 0
        @test 0 < nrow(scope3.by_category) <= 15
        @test all(1 .<= scope3.by_category.cat .<= 15)
        @test sum(scope3.by_category.E) ≈ scope3.E_Scope3
        @test sum(scope3.by_category.share) ≈ 1.0 atol=1e-9
        @test !isempty(scope3.hotspots)
        @test all(!isempty(r.method) for r in scope3.rows)
        # the hotspot list is the Pareto head
        sorted = sort(scope3.by_category, :share, rev=true)
        @test sorted.category[1] == first(scope3.hotspots)

        inv = compute_inventory(records; consolidation=:operational, date=Date(2024, 12, 31))
        @test inv.E_total ≈ inv.E_Scope1 + inv.E_Scope2_loc + inv.E_Scope3
        @test nrow(contribution_table(inv)) == 4
        @test nrow(inventory_rows(inv)) == length(inv.rows)
        @test sum(scope1_breakdown(inv).E) ≈ inv.E_Scope1
        # consolidation: equity share can only reduce a control-based total
        master = CSV.read(CarbonAccounting.datadir("reference", "master_data.csv"), DataFrame)
        shares = Dict(String(r.site) => Float64(r.equity_share) for r in eachrow(master))
        eq = compute_inventory(records; consolidation=:equity, shares=shares,
                               date=Date(2024, 12, 31))
        @test eq.E_total < inv.E_total
        @test eq.consolidation == :equity
    end

    @testset "uncertainty: GUM versus Monte Carlo" begin
        inv = compute_inventory(test_records().records; date=Date(2024, 12, 31))
        rows = total_rows(inv)                     # Scope 1 + location-based Scope 2 + Scope 3
        @test Σᵢ(r -> r.E, rows) ≈ inv.E_total
        @test length(rows) < length(inv.rows)      # the market-based rows are a parallel report
        @test all(r.u_act >= 0 && r.u_fac >= 0 for r in rows)
        @test all(r.u_rel ≈ sqrt(r.u_act^2 + r.u_fac^2) for r in rows)
        g = gum_uncertainty(rows)
        mc = monte_carlo_uncertainty(rows; n_MC=2_000, seed=42)
        @test g.E ≈ inv.E_total
        @test g.u_c > 0 && g.u_rel > 0
        @test abs(mc.u_rel - g.u_rel) / g.u_rel < 0.35        # the two routes must agree
        @test mc.lo95 < mc.E < mc.hi95
        @test mc.quantiles[1] < mc.quantiles[end]
        @test nrow(uncertainty_table(g, mc)) == 3
        @test occursin("tCO₂e", uncertainty_statement(mc))
        # correlation between factor groups *increases* the uncertainty
        @test gum_uncertainty(rows; rho_factor=0.0).u_c < g.u_c
        # the Monte Carlo can also wrap a non-linear model
        nl = monte_carlo_uncertainty(rng -> flare_emissions(1.0e6; EF_flare=1.85,
                                                            DRE=0.95 + 0.04 * rand(rng)).E;
                                     n_MC=500, seed=1)
        @test nl.E > 0 && nl.lo95 < nl.hi95
    end

    @testset "soft sensor: fit, honesty, campaign design, annualisation" begin
        campaign = CSV.read(CarbonAccounting.datadir("raw", "analyzer_campaign_2024.csv"), DataFrame)
        dcs = CSV.read(CarbonAccounting.datadir("raw", "dcs_hourly_2024.csv"), DataFrame)
        tags = ["load_pct", "stack_temp_C", "o2_pct", "flare_flow_m3h", "pressure_bar",
                "feedstock_tph"]
        X, y, X_year = Matrix(campaign[:, tags]), Vector(campaign.ch4_fraction_pct),
                       Matrix(dcs[:, tags])
        net = train_soft_sensor(X, y; hidden=[12], heteroscedastic=true, epochs=400,
                                η=0.02, seed=1)
        ŷ = predict(net, X)
        iv = predict_interval(net, X)
        m = softsensor_metrics(y, ŷ; intervals=iv)
        @test m.n == length(y)
        @test m.R² > 0.5                                      # beats a constant predictor
        @test m.RMSE > 0 && m.RMSE < std(y)
        @test 0.7 <= m.coverage95 <= 1.0                      # the interval is honest
        @test length(net.history) > 5
        @test training_history(net).val[end] <= training_history(net).val[1]
        @test nparams(net) > 0
        @test all(iv[i].lo <= iv[i].ŷ <= iv[i].hi for i in eachindex(iv))
        # cross-validation runs and reports per fold
        cv = cross_validate_softsensor(X, y; k=3, epochs=200, seed=2)
        @test nrow(cv) == 3 && all(cv.RMSE .> 0)
        # ensemble interval combines epistemic and aleatoric spread
        nets = train_ensemble(X, y; n_models=2, epochs=200, heteroscedastic=true)
        ei = ensemble_interval(nets, X)
        @test all(i.σ_total >= i.σ_epi - 1e-12 for i in ei)
        @test all(i.lo <= i.ŷ <= i.hi for i in ei)
        # the campaign does not span the year — the design check must say so
        design = campaign_design(X, X_year; feature_names=tags, activity=Vector(dcs.flare_flow_m3h))
        @test 0 <= design.coverage <= 1 && 0 <= design.risk_share <= 1
        @test design.coverage < 0.95
        @test nrow(design.features) == length(tags)
        @test min_campaign_samples(0.35; δ=0.05) == ceil(Int, (1.96 * 0.35 / 0.05)^2)
        # annualisation: the scale factor is explicit, so the physics is checkable
        ann = annualize_with_softsensor(nets, X_year, Vector(dcs.flare_flow_m3h);
                                        target_scale=CarbonAccounting.ρ_CH₄ / 100, hours=1.0)
        @test ann.E > 0 && ann.lo < ann.E < ann.hi
        @test nrow(ann.per_hour) == size(X_year, 1)
        @test annualize_with_softsensor(nets[1:1], X_year, Vector(dcs.flare_flow_m3h);
                                        target_scale=1.0).E > ann.E * 100    # scale matters
        @test occursin("tCO₂e", annualization_statement(ann))
        @test occursin("activity ratio", annualization_statement(annualize_ratio([1.0, 2.0], [1.0], 3.0)))
    end

    @testset "anomaly detection" begin
        # a synthetic step shift: CUSUM/EWMA must find it, thresholds must not cry wolf
        rng = MersenneTwister(7)
        r̂ = vcat(0.05 .* randn(rng, 300), 1.2 .+ 0.05 .* randn(rng, 30), 0.05 .* randn(rng, 200))
        ewma = ewma_chart(r̂; λ=0.2, L=3.0)
        cusum = cusum_chart(r̂; k=0.5, h=5.0)
        @test any(ewma.flags[300:360])
        @test any(cusum.flags[300:340])
        @test !any(ewma.flags[1:250])                     # in-control region stays quiet
        cal = calibrate_threshold(abs.(r̂); α=0.01)
        @test cal.α == 0.01 && cal.ARL₀ == 100
        @test count(x -> x > cal.limit, abs.(r̂)) / length(r̂) <= 0.02
        @test ARL0(0.005) == 200
        rep = detect_anomalies(r̂; α=0.01)
        @test rep isa AnomalyReport
        @test any(e -> e.method == "CUSUM" && e.t > 300, rep.events)
        @test nrow(events_dataframe(rep.events)) == length(rep.events)
        @test nrow(rep.summary) > 0
        # multivariate: an injected outlier must show up in T² and SPE
        Xn = randn(rng, 200, 4)
        Xo = vcat(Xn, [12.0, -9.0, 8.0, 11.0]')
        h = hotelling_t2(Xo; reference=Xn, α=0.01)
        @test h.T²[end] > h.limit                    # the injected point is flagged
        @test count(h.flags[1:200]) <= 5             # ≈ α·n false alarms on the reference, no more
        pc = pca_spe(Xo; k=2, reference=Xn, α=0.01)
        @test pc.SPE[end] > pc.SPE_limit
        @test 0 < pc.explained_variance <= 1
        # autoencoder + feature attribution on the DCS file
        dcs = CSV.read(CarbonAccounting.datadir("raw", "dcs_hourly_2024.csv"), DataFrame)
        tags = ["load_pct", "stack_temp_C", "o2_pct", "flare_flow_m3h", "pressure_bar",
                "feedstock_tph"]
        X = Matrix(dcs[:, tags])
        episodes = [40 * 24 .+ (1:30), 120 * 24 .+ (1:20), 210 * 24 .+ (1:24), 300 * 24 .+ (1:12)]
        ref_idx = setdiff(1:size(X, 1), vcat(episodes...))[1:5:end]
        report = detect_anomalies(X; reference=X[ref_idx, :], α=0.01, k_pca=3,
                                  ae_hidden=[8], ae_epochs=200, names=tags)
        flagged = falses(size(X, 1))
        for e in report.events
            e.severity == :low || (flagged[e.t] = true)
        end
        @test count(flagged[vcat(episodes...)]) > 50        # most injected hours are caught
        @test count(flagged) / length(flagged) < 0.25       # without drowning in false alarms
        # the autoencoder attributes the leak episode to the pressure tag
        ae = train_autoencoder(X[ref_idx, :]; hidden=[8], epochs=150, seed=3)
        attr = feature_attribution(ae, X[episodes[3], :], tags)
        @test nrow(attr) == length(tags)
        @test attr.feature[1] == "pressure_bar"
        @test sum(attr.share) ≈ 1.0
        @test length(reconstruction_error(ae, X[ref_idx, :])) == length(ref_idx)
        # the impact translation
        impact = anomaly_emission_impact(flagged, rand(rng, size(X, 1)) .+ 0.1)
        @test impact.E_excess >= 0 && 0 <= impact.share_flagged <= 1
    end

    @testset "reporting, disclosure and the submission package" begin
        inv = compute_inventory(test_records().records; consolidation=:operational,
                                date=Date(2024, 12, 31))
        dqi = dqi_table(inv.rows)
        @test nrow(dqi) > 0
        @test all(0 .<= dqi.share .<= 1)
        @test sum(dqi.E) ≈ sum(r.E for r in inv.rows)
        @test dqi_label(1.2) != dqi_label(4.0)
        inten = intensity_metrics(inv; production_t=6.53e6, revenue_usd=2.48e9, employees=2160)
        @test nrow(inten) == 6
        @test all(inten.value .> 0)
        disp = disclosure_table(inv; uncertainty=monte_carlo_uncertainty(total_rows(inv); n_MC=500),
                                intensity=inten)
        @test nrow(disp) > 5
        @test any(occursin.("Scope 1", disp.item))
        @test any(occursin.("Total", disp.item))
        stat = compare_years(inv, inv)
        @test all(isapprox.(stat.Δ, 0.0; atol=1e-9))
        pkg = export_inventory(inv; dir=CarbonAccounting.builddir("test", "submission"),
                               library=default_library(), ledger=AuditLedger())
        @test length(pkg.files) >= 7
        @test all(isfile, values(pkg.files))
        @test pkg.manifest["totals"]["total"] ≈ inv.E_total
        @test length(string(pkg.manifest["files"])) > 10       # hashes are recorded
        md = report_markdown(inv; uncertainty=gum_uncertainty(total_rows(inv)), dqi=dqi)
        @test occursin("Total", md) && occursin("tCO₂e", md)
    end

end
