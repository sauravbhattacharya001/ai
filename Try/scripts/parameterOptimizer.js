'use strict';

/**
 * Print Parameter Optimizer for BioBots
 *
 * Multi-objective optimization engine that finds optimal bioprinting parameters
 * given target outcomes. Uses grid search + refinement with Pareto front analysis
 * to balance competing objectives (cell viability vs resolution vs speed).
 *
 * Features:
 *   - 6 parameter dimensions (pressure, speed, temperature, nozzle diameter, layer height, crosslink intensity)
 *   - 5 objective functions (cell viability, resolution, speed, structural integrity, material efficiency)
 *   - Pareto front extraction for multi-objective trade-off analysis
 *   - Constraint satisfaction (min/max bounds, forbidden zones)
 *   - Material-specific presets (6 bioink types)
 *   - Sensitivity analysis showing parameter impact on objectives
 *   - Configuration comparison with improvement suggestions
 *   - Batch optimization for multiple target profiles
 *
 * Usage:
 *   const optimizer = createParameterOptimizer();
 *   const result = optimizer.optimize({ targets: { viability: 0.9, resolution: 0.8 } });
 *   const pareto = optimizer.paretoFront({ objectives: ['viability', 'resolution'] });
 *   const sensitivity = optimizer.sensitivityAnalysis(params);
 */

function createParameterOptimizer(config) {

    // ── Parameter Bounds ────────────────────────────────────────

    const DEFAULT_BOUNDS = Object.freeze({
        pressure:           Object.freeze({ min: 5,    max: 250,  unit: 'kPa',  step: 5    }),
        speed:              Object.freeze({ min: 0.5,  max: 50,   unit: 'mm/s', step: 0.5  }),
        temperature:        Object.freeze({ min: 4,    max: 42,   unit: '°C',   step: 1    }),
        nozzleDiameter:     Object.freeze({ min: 0.1,  max: 1.0,  unit: 'mm',   step: 0.05 }),
        layerHeight:        Object.freeze({ min: 0.05, max: 0.8,  unit: 'mm',   step: 0.05 }),
        crosslinkIntensity: Object.freeze({ min: 0,    max: 100,  unit: '%',    step: 5    }),
    });

    // ── Material Presets ────────────────────────────────────────

    const MATERIAL_PRESETS = Object.freeze({
        gelatin_methacrylate: Object.freeze({
            name: 'GelMA',
            bounds: { pressure: { min: 20, max: 150 }, temperature: { min: 20, max: 37 }, crosslinkIntensity: { min: 10, max: 80 } },
            weights: { viability: 1.2, resolution: 1.0, speed: 0.8, integrity: 1.0, efficiency: 1.0 },
            optimalTemp: 25,
            viscosityFactor: 0.7,
        }),
        alginate: Object.freeze({
            name: 'Alginate',
            bounds: { pressure: { min: 10, max: 120 }, temperature: { min: 20, max: 37 }, crosslinkIntensity: { min: 20, max: 100 } },
            weights: { viability: 1.0, resolution: 0.9, speed: 1.1, integrity: 1.2, efficiency: 1.0 },
            optimalTemp: 25,
            viscosityFactor: 0.5,
        }),
        collagen: Object.freeze({
            name: 'Collagen',
            bounds: { pressure: { min: 5, max: 80 }, temperature: { min: 4, max: 25 }, crosslinkIntensity: { min: 5, max: 50 } },
            weights: { viability: 1.3, resolution: 0.8, speed: 0.7, integrity: 0.9, efficiency: 1.0 },
            optimalTemp: 10,
            viscosityFactor: 0.9,
        }),
        hyaluronic_acid: Object.freeze({
            name: 'Hyaluronic Acid',
            bounds: { pressure: { min: 15, max: 130 }, temperature: { min: 20, max: 37 }, crosslinkIntensity: { min: 10, max: 70 } },
            weights: { viability: 1.1, resolution: 1.0, speed: 0.9, integrity: 1.1, efficiency: 1.0 },
            optimalTemp: 25,
            viscosityFactor: 0.8,
        }),
        fibrin: Object.freeze({
            name: 'Fibrin',
            bounds: { pressure: { min: 5, max: 60 }, temperature: { min: 20, max: 37 }, crosslinkIntensity: { min: 0, max: 30 } },
            weights: { viability: 1.4, resolution: 0.7, speed: 0.6, integrity: 0.8, efficiency: 1.1 },
            optimalTemp: 37,
            viscosityFactor: 0.3,
        }),
        pcl: Object.freeze({
            name: 'PCL (Polycaprolactone)',
            bounds: { pressure: { min: 50, max: 250 }, temperature: { min: 60, max: 120 }, crosslinkIntensity: { min: 0, max: 0 } },
            weights: { viability: 0.0, resolution: 1.3, speed: 1.0, integrity: 1.4, efficiency: 0.9 },
            optimalTemp: 80,
            viscosityFactor: 1.2,
        }),
    });

    // ── Objective Functions ──────────────────────────────────────

    const bounds = {};
    for (const [k, v] of Object.entries(DEFAULT_BOUNDS)) {
        bounds[k] = { ...v };
    }
    const materialConfig = config && config.material ? MATERIAL_PRESETS[config.material] || null : null;

    if (materialConfig && materialConfig.bounds) {
        for (const [param, overrides] of Object.entries(materialConfig.bounds)) {
            if (bounds[param]) {
                bounds[param] = { ...bounds[param], ...overrides };
            }
        }
    }

    const objectiveWeights = materialConfig ? { ...materialConfig.weights } : {
        viability: 1.0, resolution: 1.0, speed: 1.0, integrity: 1.0, efficiency: 1.0,
    };

    if (config && config.weights) {
        Object.assign(objectiveWeights, config.weights);
    }

    function clamp(val, lo, hi) {
        return Math.max(lo, Math.min(hi, val));
    }

    function normalize(val, lo, hi) {
        if (hi <= lo) return 0.5;
        return clamp((val - lo) / (hi - lo), 0, 1);
    }

    function evalViability(p) {
        const optTemp = materialConfig ? materialConfig.optimalTemp : 25;
        const tempScore = 1 - Math.min(1, Math.abs(p.temperature - optTemp) / 20);
        const pressureScore = 1 - normalize(p.pressure, 20, 200);
        const shear = (p.pressure * p.speed) / (p.nozzleDiameter * p.nozzleDiameter);
        const shearScore = 1 - normalize(shear, 0, 50000);
        const crosslinkScore = p.crosslinkIntensity <= 50
            ? 1.0
            : 1 - normalize(p.crosslinkIntensity, 50, 100) * 0.6;
        return clamp(tempScore * 0.3 + pressureScore * 0.25 + shearScore * 0.3 + crosslinkScore * 0.15, 0, 1);
    }

    function evalResolution(p) {
        const nozzleScore = 1 - normalize(p.nozzleDiameter, 0.1, 1.0);
        const layerScore = 1 - normalize(p.layerHeight, 0.05, 0.8);
        const speedScore = 1 - normalize(p.speed, 0.5, 50);
        const pressureFit = 1 - Math.abs(normalize(p.pressure, 5, 250) - 0.4) * 1.5;
        return clamp(nozzleScore * 0.35 + layerScore * 0.3 + speedScore * 0.2 + Math.max(0, pressureFit) * 0.15, 0, 1);
    }

    function evalSpeed(p) {
        const speedScore = normalize(p.speed, 0.5, 50);
        const nozzleScore = normalize(p.nozzleDiameter, 0.1, 1.0);
        const layerScore = normalize(p.layerHeight, 0.05, 0.8);
        return clamp(speedScore * 0.5 + nozzleScore * 0.25 + layerScore * 0.25, 0, 1);
    }

    function evalIntegrity(p) {
        const crosslinkScore = normalize(p.crosslinkIntensity, 0, 80);
        const layerAdhesion = 1 - normalize(p.speed, 0.5, 50) * 0.6;
        const layerFit = 1 - Math.abs(normalize(p.layerHeight, 0.05, 0.8) - 0.4);
        const tempStability = 1 - Math.abs(normalize(p.temperature, 4, 42) - 0.5) * 0.4;
        return clamp(crosslinkScore * 0.35 + layerAdhesion * 0.25 + layerFit * 0.2 + tempStability * 0.2, 0, 1);
    }

    function evalEfficiency(p) {
        const ratio = p.layerHeight / p.nozzleDiameter;
        const ratioScore = ratio >= 0.3 && ratio <= 0.8 ? 1.0 : 1 - Math.min(1, Math.abs(ratio - 0.55) * 2);
        const pressureScore = 1 - normalize(p.pressure, 80, 250) * 0.5;
        const speedScore = 1 - normalize(p.speed, 20, 50) * 0.4;
        return clamp(ratioScore * 0.4 + pressureScore * 0.3 + speedScore * 0.3, 0, 1);
    }

    const OBJECTIVE_FNS = {
        viability: evalViability,
        resolution: evalResolution,
        speed: evalSpeed,
        integrity: evalIntegrity,
        efficiency: evalEfficiency,
    };

    function evaluateAll(params) {
        const scores = {};
        for (const [name, fn] of Object.entries(OBJECTIVE_FNS)) {
            scores[name] = fn(params);
        }
        return scores;
    }

    function weightedScore(scores, targetWeights) {
        const w = targetWeights || objectiveWeights;
        let sum = 0, wSum = 0;
        for (const [k, v] of Object.entries(scores)) {
            const wt = w[k] !== undefined ? w[k] : 1;
            sum += v * wt;
            wSum += wt;
        }
        return wSum > 0 ? sum / wSum : 0;
    }

    // ── Constraints ─────────────────────────────────────────────

    const constraints = config && config.constraints ? [...config.constraints] : [];

    function satisfiesConstraints(params) {
        for (const c of constraints) {
            if (c.type === 'range') {
                const v = params[c.param];
                if (v === undefined) continue;
                if (c.min !== undefined && v < c.min) return false;
                if (c.max !== undefined && v > c.max) return false;
            } else if (c.type === 'forbidden') {
                let match = true;
                for (const [param, range] of Object.entries(c.zone)) {
                    const v = params[param];
                    if (v === undefined || v < range.min || v > range.max) { match = false; break; }
                }
                if (match) return false;
            } else if (c.type === 'ratio') {
                const a = params[c.paramA];
                const b = params[c.paramB];
                if (a !== undefined && b !== undefined && b !== 0) {
                    const ratio = a / b;
                    if (c.min !== undefined && ratio < c.min) return false;
                    if (c.max !== undefined && ratio > c.max) return false;
                }
            }
        }
        return true;
    }

    // ── Grid Search ─────────────────────────────────────────────

    function generateGrid(resolution) {
        const res = resolution || 'medium';
        const divisors = { coarse: 4, medium: 6, fine: 10 };
        const n = divisors[res] || 6;

        const paramNames = Object.keys(bounds);
        const paramSteps = {};
        for (const name of paramNames) {
            const b = bounds[name];
            const step = (b.max - b.min) / n;
            const vals = [];
            for (let i = 0; i <= n; i++) {
                vals.push(Math.round((b.min + step * i) * 1000) / 1000);
            }
            paramSteps[name] = vals;
        }
        return { paramNames, paramSteps };
    }

    function searchGrid(targetWeights, resolution, maxResults) {
        const { paramNames, paramSteps } = generateGrid(resolution);
        const max = maxResults || 20;

        const lengths = paramNames.map(n => paramSteps[n].length);
        const total = lengths.reduce((a, b) => a * b, 1);
        const limit = Math.min(total, 500000);

        const results = [];

        for (let i = 0; i < limit; i++) {
            let idx = i;
            const params = {};
            for (let d = paramNames.length - 1; d >= 0; d--) {
                const dim = paramNames[d];
                const li = lengths[d];
                params[dim] = paramSteps[dim][idx % li];
                idx = Math.floor(idx / li);
            }

            if (!satisfiesConstraints(params)) continue;

            const scores = evaluateAll(params);
            const ws = weightedScore(scores, targetWeights);

            if (results.length < max) {
                results.push({ params, scores, weightedScore: ws });
                results.sort((a, b) => b.weightedScore - a.weightedScore);
            } else if (ws > results[results.length - 1].weightedScore) {
                results[results.length - 1] = { params, scores, weightedScore: ws };
                results.sort((a, b) => b.weightedScore - a.weightedScore);
            }
        }

        return results;
    }

    function refine(candidate, targetWeights, steps) {
        const n = steps || 20;
        let best = { ...candidate.params };
        let bestScore = candidate.weightedScore;
        const paramNames = Object.keys(bounds);

        for (let iter = 0; iter < n; iter++) {
            let improved = false;
            for (const param of paramNames) {
                const b = bounds[param];
                const delta = b.step || ((b.max - b.min) / 40);
                for (const dir of [-1, 1]) {
                    const trial = { ...best };
                    trial[param] = clamp(trial[param] + dir * delta, b.min, b.max);
                    trial[param] = Math.round(trial[param] * 1000) / 1000;
                    if (!satisfiesConstraints(trial)) continue;
                    const scores = evaluateAll(trial);
                    const ws = weightedScore(scores, targetWeights);
                    if (ws > bestScore) {
                        best = trial;
                        bestScore = ws;
                        improved = true;
                    }
                }
            }
            if (!improved) break;
        }

        const scores = evaluateAll(best);
        return { params: best, scores, weightedScore: bestScore };
    }

    // ── Pareto Front ────────────────────────────────────────────

    function dominates(a, b, objectives) {
        let dominated = false;
        for (const obj of objectives) {
            if (a.scores[obj] < b.scores[obj]) return false;
            if (a.scores[obj] > b.scores[obj]) dominated = true;
        }
        return dominated;
    }

    function extractParetoFront(candidates, objectives) {
        const front = [];
        for (const c of candidates) {
            let dominated = false;
            for (const f of front) {
                if (dominates(f, c, objectives)) { dominated = true; break; }
            }
            if (!dominated) {
                for (let i = front.length - 1; i >= 0; i--) {
                    if (dominates(c, front[i], objectives)) front.splice(i, 1);
                }
                front.push(c);
            }
        }
        return front;
    }

    // ── Public API ──────────────────────────────────────────────

    function optimize(options) {
        const opts = options || {};
        const targetWeights = opts.targets || objectiveWeights;
        const resolution = opts.resolution || 'medium';
        const topN = opts.topN || 10;

        const gridResults = searchGrid(targetWeights, resolution, topN * 3);
        const refined = gridResults.map(r => refine(r, targetWeights));
        refined.sort((a, b) => b.weightedScore - a.weightedScore);

        const deduped = [];
        for (const r of refined) {
            let isDup = false;
            for (const d of deduped) {
                const diff = Object.keys(r.params).reduce((s, k) => s + Math.abs(r.params[k] - d.params[k]), 0);
                if (diff < 1) { isDup = true; break; }
            }
            if (!isDup) deduped.push(r);
            if (deduped.length >= topN) break;
        }

        const best = deduped[0] || null;
        return {
            best,
            alternatives: deduped.slice(1),
            totalEvaluated: gridResults.length,
            material: materialConfig ? materialConfig.name : 'Generic',
            recommendation: best ? buildRecommendation(best) : 'No feasible solution found.',
        };
    }

    function paretoFront(options) {
        const opts = options || {};
        const objectives = opts.objectives || ['viability', 'resolution'];
        const resolution = opts.resolution || 'medium';

        if (objectives.length < 2) {
            return { error: 'Pareto front requires at least 2 objectives.', front: [] };
        }

        const gridResults = searchGrid(null, resolution, 200);
        const refined = gridResults.map(r => refine(r, null, 10));
        const front = extractParetoFront(refined, objectives);

        front.sort((a, b) => b.scores[objectives[0]] - a.scores[objectives[0]]);

        return {
            objectives,
            front,
            size: front.length,
            extremes: objectives.reduce((acc, obj) => {
                const sorted = [...front].sort((a, b) => b.scores[obj] - a.scores[obj]);
                acc[obj] = { best: sorted[0] || null, worst: sorted[sorted.length - 1] || null };
                return acc;
            }, {}),
        };
    }

    function sensitivityAnalysis(baseParams) {
        const base = baseParams || getDefaultParams();
        const baseScores = evaluateAll(base);
        const paramNames = Object.keys(bounds);
        const analysis = {};

        for (const param of paramNames) {
            const b = bounds[param];
            const range = b.max - b.min;
            const perturbations = [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2];
            const impacts = {};

            for (const objName of Object.keys(OBJECTIVE_FNS)) {
                let totalImpact = 0;
                let count = 0;

                for (const pct of perturbations) {
                    const trial = { ...base };
                    trial[param] = clamp(base[param] + pct * range, b.min, b.max);
                    const trialScores = evaluateAll(trial);
                    totalImpact += Math.abs(trialScores[objName] - baseScores[objName]);
                    count++;
                }

                impacts[objName] = Math.round((totalImpact / count) * 10000) / 10000;
            }

            const avgImpact = Object.values(impacts).reduce((s, v) => s + v, 0) / Object.values(impacts).length;
            analysis[param] = { impacts, averageImpact: Math.round(avgImpact * 10000) / 10000 };
        }

        const ranked = Object.entries(analysis)
            .sort((a, b) => b[1].averageImpact - a[1].averageImpact)
            .map(([param, data], i) => ({ rank: i + 1, param, ...data }));

        return {
            baseParams: base,
            baseScores,
            parameters: ranked,
            mostInfluential: ranked[0] ? ranked[0].param : null,
            leastInfluential: ranked[ranked.length - 1] ? ranked[ranked.length - 1].param : null,
        };
    }

    function compareConfigurations(paramsA, paramsB) {
        const scoresA = evaluateAll(paramsA);
        const scoresB = evaluateAll(paramsB);
        const wsA = weightedScore(scoresA);
        const wsB = weightedScore(scoresB);

        const comparison = {};
        for (const obj of Object.keys(OBJECTIVE_FNS)) {
            const diff = scoresB[obj] - scoresA[obj];
            comparison[obj] = {
                configA: Math.round(scoresA[obj] * 1000) / 1000,
                configB: Math.round(scoresB[obj] * 1000) / 1000,
                difference: Math.round(diff * 1000) / 1000,
                winner: diff > 0.01 ? 'B' : diff < -0.01 ? 'A' : 'tie',
            };
        }

        const suggestions = [];
        if (scoresA.viability < 0.5) suggestions.push('Config A: reduce pressure or speed to improve cell viability.');
        if (scoresB.viability < 0.5) suggestions.push('Config B: reduce pressure or speed to improve cell viability.');
        if (scoresA.resolution < 0.5) suggestions.push('Config A: use smaller nozzle or thinner layers for better resolution.');
        if (scoresB.resolution < 0.5) suggestions.push('Config B: use smaller nozzle or thinner layers for better resolution.');

        return {
            configA: { params: paramsA, scores: scoresA, weightedScore: Math.round(wsA * 1000) / 1000 },
            configB: { params: paramsB, scores: scoresB, weightedScore: Math.round(wsB * 1000) / 1000 },
            comparison,
            overallWinner: wsA > wsB + 0.01 ? 'A' : wsB > wsA + 0.01 ? 'B' : 'tie',
            suggestions,
        };
    }

    function batchOptimize(profiles) {
        return profiles.map((profile, i) => {
            const result = optimize(profile);
            return { profileIndex: i, name: profile.name || ('Profile ' + (i + 1)), ...result };
        });
    }

    function evaluateParams(params) {
        const scores = evaluateAll(params);
        const ws = weightedScore(scores);
        return {
            params,
            scores,
            weightedScore: Math.round(ws * 1000) / 1000,
            feasible: satisfiesConstraints(params),
            recommendation: buildRecommendation({ params, scores, weightedScore: ws }),
        };
    }

    function getDefaultParams() {
        const defaults = {};
        for (const [param, b] of Object.entries(bounds)) {
            defaults[param] = Math.round(((b.min + b.max) / 2) * 1000) / 1000;
        }
        return defaults;
    }

    function buildRecommendation(result) {
        const lines = [];
        const s = result.scores;

        if (result.weightedScore >= 0.75) {
            lines.push('Excellent parameter set — well balanced across objectives.');
        } else if (result.weightedScore >= 0.55) {
            lines.push('Good parameter set with room for improvement.');
        } else {
            lines.push('Suboptimal parameters — consider adjusting.');
        }

        if (s.viability < 0.5) lines.push('⚠ Low cell viability: reduce pressure/speed or adjust temperature closer to optimal.');
        if (s.resolution < 0.4) lines.push('⚠ Low resolution: use smaller nozzle diameter and thinner layers.');
        if (s.speed < 0.3) lines.push('ℹ Low throughput: increase speed or use larger nozzle if viability allows.');
        if (s.integrity < 0.4) lines.push('⚠ Structural risk: increase crosslink intensity or reduce print speed.');
        if (s.efficiency < 0.4) lines.push('ℹ Material waste: adjust layer height to 30-80% of nozzle diameter.');

        return lines.join(' ');
    }

    function getMaterialPresets() {
        return Object.entries(MATERIAL_PRESETS).map(([key, preset]) => ({
            key,
            name: preset.name,
            bounds: preset.bounds,
            optimalTemp: preset.optimalTemp,
        }));
    }

    function getBounds() {
        return { ...bounds };
    }

    function addConstraint(constraint) {
        if (!constraint || !constraint.type) return false;
        constraints.push(constraint);
        return true;
    }

    function getConstraints() {
        return [...constraints];
    }

    function generateReport(result) {
        const lines = [];
        lines.push('╔══════════════════════════════════════════════════════╗');
        lines.push('║         PRINT PARAMETER OPTIMIZATION REPORT         ║');
        lines.push('╚══════════════════════════════════════════════════════╝');
        lines.push('');
        lines.push('Material: ' + (materialConfig ? materialConfig.name : 'Generic'));
        lines.push('');

        if (result.best) {
            lines.push('── Best Parameters ────────────────────────────────────');
            for (const [k, v] of Object.entries(result.best.params)) {
                const b = bounds[k];
                lines.push('  ' + k.padEnd(22) + ': ' + v + ' ' + (b ? b.unit : ''));
            }
            lines.push('');
            lines.push('── Objective Scores ───────────────────────────────────');
            for (const [k, v] of Object.entries(result.best.scores)) {
                const pct = (v * 100).toFixed(1);
                const bar = '█'.repeat(Math.round(v * 20)) + '░'.repeat(20 - Math.round(v * 20));
                lines.push('  ' + k.padEnd(14) + ': ' + bar + ' ' + pct + '%');
            }
            lines.push('');
            lines.push('  Weighted Score: ' + (result.best.weightedScore * 100).toFixed(1) + '%');
            lines.push('');
            lines.push('── Recommendation ─────────────────────────────────────');
            lines.push('  ' + result.recommendation);
        } else {
            lines.push('  No feasible solution found.');
        }

        if (result.alternatives && result.alternatives.length > 0) {
            lines.push('');
            lines.push('── Alternatives (' + result.alternatives.length + ') ──────────────────────────');
            for (let i = 0; i < Math.min(3, result.alternatives.length); i++) {
                const alt = result.alternatives[i];
                lines.push('  #' + (i + 2) + ' score: ' + (alt.weightedScore * 100).toFixed(1) + '%');
            }
        }

        return lines.join('\n');
    }

    return Object.freeze({
        optimize,
        paretoFront,
        sensitivityAnalysis,
        compareConfigurations,
        batchOptimize,
        evaluateParams,
        getDefaultParams,
        getMaterialPresets,
        getBounds,
        addConstraint,
        getConstraints,
        generateReport,
    });
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { createParameterOptimizer };
}
