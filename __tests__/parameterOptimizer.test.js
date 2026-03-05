'use strict';

const { createParameterOptimizer } = require('../Try/scripts/parameterOptimizer');

describe('createParameterOptimizer', () => {
    let optimizer;

    beforeEach(() => {
        optimizer = createParameterOptimizer();
    });

    test('creates optimizer with default config', () => {
        expect(optimizer).toBeDefined();
        expect(typeof optimizer.optimize).toBe('function');
        expect(typeof optimizer.paretoFront).toBe('function');
        expect(typeof optimizer.sensitivityAnalysis).toBe('function');
    });

    test('creates optimizer with material preset', () => {
        const opt = createParameterOptimizer({ material: 'alginate' });
        expect(opt).toBeDefined();
    });

    test('creates optimizer with custom weights', () => {
        const opt = createParameterOptimizer({ weights: { viability: 2.0, speed: 0.5 } });
        expect(opt).toBeDefined();
    });

    test('handles unknown material gracefully', () => {
        const opt = createParameterOptimizer({ material: 'unknown_stuff' });
        expect(opt).toBeDefined();
    });

    test('getBounds returns all 6 parameters', () => {
        const b = optimizer.getBounds();
        expect(Object.keys(b)).toHaveLength(6);
        expect(b.pressure).toBeDefined();
        expect(b.nozzleDiameter).toBeDefined();
    });

    test('getBounds reflects material overrides', () => {
        const opt = createParameterOptimizer({ material: 'collagen' });
        const b = opt.getBounds();
        expect(b.pressure.max).toBe(80);
        expect(b.temperature.max).toBe(25);
    });

    test('getDefaultParams returns midpoints', () => {
        const p = optimizer.getDefaultParams();
        expect(p.pressure).toBeGreaterThan(0);
        expect(p.speed).toBeGreaterThan(0);
    });

    test('getMaterialPresets returns 6 presets', () => {
        const presets = optimizer.getMaterialPresets();
        expect(presets).toHaveLength(6);
        expect(presets[0].name).toBeDefined();
    });

    test('evaluateParams returns scores for given params', () => {
        const result = optimizer.evaluateParams({
            pressure: 50, speed: 5, temperature: 25,
            nozzleDiameter: 0.4, layerHeight: 0.2, crosslinkIntensity: 30,
        });
        expect(result.scores).toBeDefined();
        expect(result.weightedScore).toBeGreaterThan(0);
        expect(result.feasible).toBe(true);
        expect(typeof result.recommendation).toBe('string');
    });

    test('scores are between 0 and 1', () => {
        const result = optimizer.evaluateParams({
            pressure: 100, speed: 10, temperature: 30,
            nozzleDiameter: 0.5, layerHeight: 0.3, crosslinkIntensity: 50,
        });
        for (const v of Object.values(result.scores)) {
            expect(v).toBeGreaterThanOrEqual(0);
            expect(v).toBeLessThanOrEqual(1);
        }
    });

    test('low pressure + slow speed gives high viability', () => {
        const result = optimizer.evaluateParams({
            pressure: 20, speed: 1, temperature: 25,
            nozzleDiameter: 0.5, layerHeight: 0.2, crosslinkIntensity: 20,
        });
        expect(result.scores.viability).toBeGreaterThan(0.6);
    });

    test('small nozzle + thin layers gives high resolution', () => {
        const result = optimizer.evaluateParams({
            pressure: 50, speed: 2, temperature: 25,
            nozzleDiameter: 0.15, layerHeight: 0.08, crosslinkIntensity: 30,
        });
        expect(result.scores.resolution).toBeGreaterThan(0.6);
    });

    test('high speed + large nozzle gives high speed score', () => {
        const result = optimizer.evaluateParams({
            pressure: 80, speed: 40, temperature: 25,
            nozzleDiameter: 0.9, layerHeight: 0.7, crosslinkIntensity: 30,
        });
        expect(result.scores.speed).toBeGreaterThan(0.7);
    });

    test('high crosslink gives high integrity', () => {
        const result = optimizer.evaluateParams({
            pressure: 50, speed: 3, temperature: 25,
            nozzleDiameter: 0.4, layerHeight: 0.2, crosslinkIntensity: 70,
        });
        expect(result.scores.integrity).toBeGreaterThan(0.5);
    });

    test('optimize returns best and alternatives', () => {
        const result = optimizer.optimize({ resolution: 'coarse' });
        expect(result.best).toBeDefined();
        expect(result.best.params).toBeDefined();
        expect(result.best.scores).toBeDefined();
        expect(result.best.weightedScore).toBeGreaterThan(0);
        expect(result.alternatives).toBeDefined();
        expect(result.material).toBe('Generic');
    });

    test('optimize with viability emphasis', () => {
        const result = optimizer.optimize({
            targets: { viability: 3.0, resolution: 0.5, speed: 0.5, integrity: 0.5, efficiency: 0.5 },
            resolution: 'coarse',
        });
        expect(result.best.scores.viability).toBeGreaterThan(0.4);
    });

    test('optimize with speed emphasis', () => {
        const result = optimizer.optimize({
            targets: { viability: 0.5, resolution: 0.5, speed: 3.0, integrity: 0.5, efficiency: 0.5 },
            resolution: 'coarse',
        });
        expect(result.best.scores.speed).toBeGreaterThan(0.4);
    });

    test('optimize with material preset', () => {
        const opt = createParameterOptimizer({ material: 'alginate' });
        const result = opt.optimize({ resolution: 'coarse' });
        expect(result.best).toBeDefined();
        expect(result.material).toBe('Alginate');
    });

    test('optimize respects topN', () => {
        const result = optimizer.optimize({ resolution: 'coarse', topN: 3 });
        expect(result.best).toBeDefined();
        expect(result.alternatives.length).toBeLessThanOrEqual(2);
    });

    test('optimize recommendation is a string', () => {
        const result = optimizer.optimize({ resolution: 'coarse' });
        expect(typeof result.recommendation).toBe('string');
        expect(result.recommendation.length).toBeGreaterThan(0);
    });

    test('paretoFront returns front for two objectives', () => {
        const result = optimizer.paretoFront({
            objectives: ['viability', 'speed'],
            resolution: 'coarse',
        });
        expect(result.front.length).toBeGreaterThan(0);
        expect(result.objectives).toEqual(['viability', 'speed']);
        expect(result.extremes.viability).toBeDefined();
    });

    test('paretoFront with single objective returns error', () => {
        const result = optimizer.paretoFront({ objectives: ['viability'] });
        expect(result.error).toBeDefined();
    });

    test('paretoFront entries are non-dominated', () => {
        const result = optimizer.paretoFront({
            objectives: ['viability', 'resolution'],
            resolution: 'coarse',
        });
        const front = result.front;
        for (let i = 0; i < front.length; i++) {
            for (let j = 0; j < front.length; j++) {
                if (i === j) continue;
                const aDom = front[i].scores.viability >= front[j].scores.viability &&
                    front[i].scores.resolution >= front[j].scores.resolution &&
                    (front[i].scores.viability > front[j].scores.viability || front[i].scores.resolution > front[j].scores.resolution);
                expect(aDom).toBe(false);
            }
        }
    });

    test('paretoFront with three objectives', () => {
        const result = optimizer.paretoFront({
            objectives: ['viability', 'resolution', 'speed'],
            resolution: 'coarse',
        });
        expect(result.front.length).toBeGreaterThan(0);
        expect(result.size).toBe(result.front.length);
    });

    test('sensitivityAnalysis returns ranked parameters', () => {
        const result = optimizer.sensitivityAnalysis();
        expect(result.parameters).toBeDefined();
        expect(result.parameters.length).toBe(6);
        expect(result.parameters[0].rank).toBe(1);
        expect(result.mostInfluential).toBeDefined();
        expect(result.leastInfluential).toBeDefined();
    });

    test('sensitivityAnalysis with custom base params', () => {
        const result = optimizer.sensitivityAnalysis({
            pressure: 50, speed: 5, temperature: 25,
            nozzleDiameter: 0.4, layerHeight: 0.2, crosslinkIntensity: 30,
        });
        expect(result.baseParams.pressure).toBe(50);
        expect(result.baseScores).toBeDefined();
    });

    test('sensitivity impacts are non-negative', () => {
        const result = optimizer.sensitivityAnalysis();
        for (const param of result.parameters) {
            expect(param.averageImpact).toBeGreaterThanOrEqual(0);
            for (const v of Object.values(param.impacts)) {
                expect(v).toBeGreaterThanOrEqual(0);
            }
        }
    });

    test('compareConfigurations returns comparison', () => {
        const a = { pressure: 30, speed: 3, temperature: 25, nozzleDiameter: 0.3, layerHeight: 0.15, crosslinkIntensity: 30 };
        const b = { pressure: 100, speed: 20, temperature: 30, nozzleDiameter: 0.8, layerHeight: 0.5, crosslinkIntensity: 60 };
        const result = optimizer.compareConfigurations(a, b);
        expect(result.configA).toBeDefined();
        expect(result.configB).toBeDefined();
        expect(result.comparison).toBeDefined();
        expect(['A', 'B', 'tie']).toContain(result.overallWinner);
    });

    test('comparison has winner per objective', () => {
        const a = { pressure: 20, speed: 1, temperature: 25, nozzleDiameter: 0.2, layerHeight: 0.1, crosslinkIntensity: 20 };
        const b = { pressure: 150, speed: 30, temperature: 35, nozzleDiameter: 0.8, layerHeight: 0.6, crosslinkIntensity: 70 };
        const result = optimizer.compareConfigurations(a, b);
        for (const obj of Object.values(result.comparison)) {
            expect(['A', 'B', 'tie']).toContain(obj.winner);
        }
    });

    test('comparing identical configs yields tie', () => {
        const params = { pressure: 50, speed: 5, temperature: 25, nozzleDiameter: 0.4, layerHeight: 0.2, crosslinkIntensity: 30 };
        const result = optimizer.compareConfigurations(params, params);
        expect(result.overallWinner).toBe('tie');
    });

    test('batchOptimize handles multiple profiles', () => {
        const profiles = [
            { name: 'High Viability', targets: { viability: 3, resolution: 1, speed: 1, integrity: 1, efficiency: 1 }, resolution: 'coarse' },
            { name: 'High Speed', targets: { viability: 1, resolution: 1, speed: 3, integrity: 1, efficiency: 1 }, resolution: 'coarse' },
        ];
        const results = optimizer.batchOptimize(profiles);
        expect(results).toHaveLength(2);
        expect(results[0].name).toBe('High Viability');
        expect(results[1].name).toBe('High Speed');
    });

    test('batchOptimize with empty array', () => {
        expect(optimizer.batchOptimize([])).toHaveLength(0);
    });

    test('addConstraint adds range constraint', () => {
        expect(optimizer.addConstraint({ type: 'range', param: 'pressure', min: 30, max: 80 })).toBe(true);
        expect(optimizer.getConstraints()).toHaveLength(1);
    });

    test('addConstraint rejects invalid', () => {
        expect(optimizer.addConstraint(null)).toBe(false);
        expect(optimizer.addConstraint({})).toBe(false);
    });

    test('range constraints affect optimization', () => {
        const opt = createParameterOptimizer({
            constraints: [{ type: 'range', param: 'pressure', min: 10, max: 30 }],
        });
        const result = opt.optimize({ resolution: 'coarse' });
        if (result.best) {
            expect(result.best.params.pressure).toBeLessThanOrEqual(30);
            expect(result.best.params.pressure).toBeGreaterThanOrEqual(10);
        }
    });

    test('forbidden zone constraint works', () => {
        const opt = createParameterOptimizer({
            constraints: [{
                type: 'forbidden',
                zone: { pressure: { min: 90, max: 200 }, speed: { min: 20, max: 50 } },
            }],
        });
        const result = opt.optimize({ resolution: 'coarse' });
        if (result.best) {
            const p = result.best.params;
            const inZone = p.pressure >= 90 && p.pressure <= 200 && p.speed >= 20 && p.speed <= 50;
            expect(inZone).toBe(false);
        }
    });

    test('ratio constraint works', () => {
        const opt = createParameterOptimizer({
            constraints: [{ type: 'ratio', paramA: 'layerHeight', paramB: 'nozzleDiameter', min: 0.3, max: 0.7 }],
        });
        const result = opt.optimize({ resolution: 'coarse' });
        if (result.best) {
            const ratio = result.best.params.layerHeight / result.best.params.nozzleDiameter;
            expect(ratio).toBeGreaterThanOrEqual(0.29);
            expect(ratio).toBeLessThanOrEqual(0.71);
        }
    });

    test('generateReport produces text output', () => {
        const result = optimizer.optimize({ resolution: 'coarse' });
        const report = optimizer.generateReport(result);
        expect(typeof report).toBe('string');
        expect(report).toContain('PRINT PARAMETER OPTIMIZATION REPORT');
        expect(report).toContain('Best Parameters');
    });

    test('generateReport with no solution', () => {
        const report = optimizer.generateReport({ best: null, recommendation: 'No feasible solution found.' });
        expect(report).toContain('No feasible solution');
    });

    test('PCL preset has zero crosslink range', () => {
        const opt = createParameterOptimizer({ material: 'pcl' });
        const b = opt.getBounds();
        expect(b.crosslinkIntensity.min).toBe(0);
        expect(b.crosslinkIntensity.max).toBe(0);
    });

    test('fibrin preset works', () => {
        const opt = createParameterOptimizer({ material: 'fibrin' });
        const result = opt.optimize({ resolution: 'coarse' });
        expect(result.material).toBe('Fibrin');
    });

    test('collagen preset has low temp bounds', () => {
        const opt = createParameterOptimizer({ material: 'collagen' });
        expect(opt.getBounds().temperature.max).toBe(25);
    });

    test('GelMA preset works', () => {
        const opt = createParameterOptimizer({ material: 'gelatin_methacrylate' });
        const result = opt.optimize({ resolution: 'coarse' });
        expect(result.material).toBe('GelMA');
    });

    test('hyaluronic acid preset works', () => {
        const opt = createParameterOptimizer({ material: 'hyaluronic_acid' });
        const result = opt.optimize({ resolution: 'coarse' });
        expect(result.material).toBe('Hyaluronic Acid');
    });

    test('extreme low values evaluate without error', () => {
        const result = optimizer.evaluateParams({
            pressure: 5, speed: 0.5, temperature: 4,
            nozzleDiameter: 0.1, layerHeight: 0.05, crosslinkIntensity: 0,
        });
        expect(result.weightedScore).toBeGreaterThanOrEqual(0);
    });

    test('extreme high values evaluate without error', () => {
        const result = optimizer.evaluateParams({
            pressure: 250, speed: 50, temperature: 42,
            nozzleDiameter: 1.0, layerHeight: 0.8, crosslinkIntensity: 100,
        });
        expect(result.weightedScore).toBeGreaterThanOrEqual(0);
    });

    test('frozen return object cannot be mutated', () => {
        expect(() => { optimizer.newProp = 'test'; }).toThrow();
    });

    test('viability and speed trade off', () => {
        const slow = optimizer.evaluateParams({
            pressure: 20, speed: 1, temperature: 25,
            nozzleDiameter: 0.4, layerHeight: 0.2, crosslinkIntensity: 20,
        });
        const fast = optimizer.evaluateParams({
            pressure: 80, speed: 40, temperature: 25,
            nozzleDiameter: 0.8, layerHeight: 0.6, crosslinkIntensity: 20,
        });
        expect(slow.scores.viability).toBeGreaterThan(fast.scores.viability);
        expect(fast.scores.speed).toBeGreaterThan(slow.scores.speed);
    });

    test('resolution and speed trade off', () => {
        const precise = optimizer.evaluateParams({
            pressure: 50, speed: 2, temperature: 25,
            nozzleDiameter: 0.15, layerHeight: 0.08, crosslinkIntensity: 30,
        });
        const fast = optimizer.evaluateParams({
            pressure: 80, speed: 35, temperature: 25,
            nozzleDiameter: 0.8, layerHeight: 0.6, crosslinkIntensity: 30,
        });
        expect(precise.scores.resolution).toBeGreaterThan(fast.scores.resolution);
        expect(fast.scores.speed).toBeGreaterThan(precise.scores.speed);
    });

    test('multiple constraints can be combined', () => {
        const opt = createParameterOptimizer({
            constraints: [
                { type: 'range', param: 'pressure', min: 20, max: 80 },
                { type: 'range', param: 'speed', min: 1, max: 10 },
            ],
        });
        const result = opt.optimize({ resolution: 'coarse' });
        if (result.best) {
            expect(result.best.params.pressure).toBeGreaterThanOrEqual(20);
            expect(result.best.params.pressure).toBeLessThanOrEqual(80);
            expect(result.best.params.speed).toBeGreaterThanOrEqual(1);
            expect(result.best.params.speed).toBeLessThanOrEqual(10);
        }
    });

    test('paretoFront extremes have best and worst', () => {
        const result = optimizer.paretoFront({
            objectives: ['viability', 'speed'],
            resolution: 'coarse',
        });
        expect(result.extremes.viability.best).toBeDefined();
        expect(result.extremes.viability.worst).toBeDefined();
        expect(result.extremes.speed.best).toBeDefined();
    });

    test('optimize with no options uses defaults', () => {
        const result = optimizer.optimize();
        expect(result.best).toBeDefined();
    });

    test('sensitivityAnalysis mostInfluential is a string', () => {
        const result = optimizer.sensitivityAnalysis();
        expect(typeof result.mostInfluential).toBe('string');
        expect(typeof result.leastInfluential).toBe('string');
    });

    test('report contains material name', () => {
        const opt = createParameterOptimizer({ material: 'alginate' });
        const result = opt.optimize({ resolution: 'coarse' });
        const report = opt.generateReport(result);
        expect(report).toContain('Alginate');
    });

    test('report contains alternative count', () => {
        const result = optimizer.optimize({ resolution: 'coarse' });
        const report = optimizer.generateReport(result);
        if (result.alternatives.length > 0) {
            expect(report).toContain('Alternatives');
        }
    });
});
