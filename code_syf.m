clearvars;
clc;
close all;

%% ============================================================
%%  ML classifiers vs Lex-PL, and three active learning strategies
%%
%%  Table 1: static classification comparison
%%     CART, KNN, SVM, RandomForest, Lex-PL
%%
%%  Table 2: active learning strategy comparison based on Lex-PL
%%     AL-Rand, AL-Entropy, TWD-AL
%%
%%  Requirements:
%%     1) Optimization Toolbox: needed by Lex-PL / TWD-AL
%%     2) Statistics and Machine Learning Toolbox: needed by CART/KNN/SVM/RF
%%
%%  Data format:
%%     Alternative(optional), g1,g2,..., Class(Cl1/Cl2/... or numeric)
%% ============================================================

cfg = makeMLActiveComparisonConfig();
validateComparisonConfig(cfg);

StaticResults = emptyResultRows();
ActiveResults = emptyResultRows();

fprintf('\n============================================================\n');
fprintf('表1：机器学习分类模型与 Lex-PL 的静态分类性能比较\n');
fprintf('表2：三种主动学习策略的效果比较\n');
fprintf('============================================================\n');
fprintf('数据目录：%s\n', cfg.dataDir);
fprintf('重复次数：%d，训练比例：%.2f，L=%d，K=%d，Ns=%d\n', ...
    cfg.nRuns, cfg.trainRatio, cfg.L, cfg.K, cfg.Ns);
fprintf('TWD参数：xi=%.2f，MAC=%.2f\n', cfg.xi, cfg.MAC);

for d = 1:numel(cfg.dataFiles)
    fileName = cfg.dataFiles{d};
    filePath = resolveDataFilePath(cfg.dataDir, fileName);
    [~, dataName, ~] = fileparts(fileName);

    fprintf('\n------------------------------------------------------------\n');
    fprintf('数据集：%s\n', dataName);
    fprintf('------------------------------------------------------------\n');

    [X, y, ~, ~] = readMCSDataset(filePath);
    q = max(y);
    validateDatasetForComparison(X, y, q, fileName);
    fprintf('样本数=%d，准则数=%d，类别数=%d\n', size(X,1), size(X,2), q);

    for run = 1:cfg.nRuns
        fprintf('  Run %02d / %02d\n', run, cfg.nRuns);
        rng(cfg.seed + 1000 * d + run, 'twister');
        [trainIdx, testIdx] = splitReferenceAndPool(y, cfg);

        Xtrain = X(trainIdx, :);
        ytrain = y(trainIdx);
        Xtest = X(testIdx, :);
        ytest = y(testIdx);

        staticRows = runStaticClassifierComparison(Xtrain, ytrain, Xtest, ytest, q, cfg, dataName, run);
        activeRows = runActiveStrategyComparison(Xtrain, ytrain, Xtest, ytest, q, cfg, dataName, run, d);

        StaticResults = [StaticResults, staticRows]; %#ok<AGROW>
        ActiveResults = [ActiveResults, activeRows]; %#ok<AGROW>
    end
end

StaticRawResults = struct2table(StaticResults);
ActiveRawResults = struct2table(ActiveResults);

staticOrder = ["CART", "KNN", "SVM", "RandomForest", "Lex-PL"];
activeOrder = ["AL-Rand", "AL-Entropy", "TWD-AL"];

StaticSummary = summarizeByMethods(StaticRawResults, staticOrder);
ActiveSummary = summarizeByMethods(ActiveRawResults, activeOrder);

fprintf('\n============================================================\n');
fprintf('表1：机器学习分类模型与 Lex-PL 的静态分类性能比较\n');
fprintf('============================================================\n');
disp(StaticSummary);

fprintf('\n============================================================\n');
fprintf('表2：三种主动学习策略的效果比较\n');
fprintf('============================================================\n');
disp(ActiveSummary);

if cfg.saveResults
    if ~exist(cfg.outputDir, 'dir')
        mkdir(cfg.outputDir);
    end
    writetable(StaticRawResults, fullfile(cfg.outputDir, 'table1_static_raw.csv'));
    writetable(ActiveRawResults, fullfile(cfg.outputDir, 'table2_active_raw.csv'));
    writetable(StaticSummary, fullfile(cfg.outputDir, 'table1_static_summary.csv'));
    writetable(ActiveSummary, fullfile(cfg.outputDir, 'table2_active_summary.csv'));
    fprintf('\n结果已保存至：%s\n', cfg.outputDir);
end

fprintf('\n程序运行完成。\n');

%% ============================================================
%% Main experiment functions
%% ============================================================

function cfg = makeMLActiveComparisonConfig()
    cfg = struct();

    % 修改为本地数据目录；若数据在当前目录或 ./data 中，可保持 pwd。
    cfg.dataDir = pwd;
    cfg.dataFiles = {
        'BCC.csv'
        'CEV.csv'
        'CPU.csv'
        'DBS.csv'
        'ERA.csv'
        'ESL.csv'
        'LEV.csv'
        'MMG.csv'
        'MPG.csv'
    };

    % 基本实验参数
    cfg.trainRatio = 0.30;
    cfg.nRuns = 10;
    cfg.numSegments = 3;
    cfg.L = cfg.numSegments;
    cfg.seed = 2;
    cfg.randomSeed = cfg.seed;
    cfg.splitMode = "Random";

    % 主动学习参数
    cfg.queryBudget = 3;
    cfg.K = cfg.queryBudget;
    cfg.query.batchSize = cfg.K;

    % 三支决策风险参数
    cfg.xi = 0.05;
    cfg.MAC = 0.95;
    cfg.thresholdMinGap = 1e-6;

    % 兼容模型采样参数
    cfg.samplingMethod = "RandomObjectiveMix";
    cfg.Ns = 100;
    cfg.objectiveSlack = 0;
    cfg.randomObjectivePoolSize = 500;
    cfg.randomObjectiveMixSize = 5;
    cfg.randomObjectiveFeasTol = 1e-7;
    cfg.randomObjectiveMaxTrialsFactor = 10;
    cfg.randomObjectiveRandomizeE = false;
    cfg.randomObjectiveEJitter = 0;
    cfg.caiPositiveTol = 1e-12;
    cfg.msPositiveTol = cfg.caiPositiveTol;
    % 为保证三种主动学习策略预算可比，TWD-AL 在 NEG 不足 K 时从剩余样本中按低支持度补足。
    % 若希望严格按照三支决策只询问 NEG，可设为 false。
    cfg.twdFillToBudget = true;

    % 机器学习模型参数
    cfg.rfNumTrees = 100;
    cfg.knnK = 5;

    % 求解器参数
    cfg.solverDisplay = 'none';
    cfg.solverAlgorithm = 'dual-simplex';
    cfg.linprogAlgorithms = [string(cfg.solverAlgorithm), "interior-point", "default"];
    cfg.linprogRelaxedTolerances = [NaN, 1e-5];
    cfg.linprogVerboseFallback = false;
    cfg.saveLinprogFailure = false;
    cfg.linprogFailureDir = pwd;

    % 是否保存结果
    cfg.saveResults = true;
    cfg.outputDir = fullfile(pwd, 'ml_active_results');
end

function rows = runStaticClassifierComparison(Xtrain, ytrain, Xtest, ytest, q, cfg, dataName, run)
    rows = emptyResultRows();
    methods = ["CART", "KNN", "SVM", "RandomForest", "Lex-PL"];

    for m = 1:numel(methods)
        methodName = methods(m);
        tStart = tic;
        try
            switch methodName
                case "Lex-PL"
                    model = trainLexPLWeighted(Xtrain, ytrain, ones(numel(ytrain), 1), q, cfg);
                    pred = predictLexPL(model, Xtest, cfg);
                otherwise
                    pred = trainPredictMLClassifier(Xtrain, ytrain, Xtest, q, methodName, cfg);
            end
            metrics = calculateOrderedMetrics(ytest, pred, q);
            evalResult = makeEvalResult(metrics, 0, 0, toc(tStart), "Static");
        catch ME
            warning('Static method %s failed on %s run %d: %s', methodName, dataName, run, ME.message);
            evalResult = makeEvalResult(makeNanMetrics(), 0, 0, toc(tStart), "Failed");
        end
        rows(end+1) = makeResultRow(dataName, run, methodName, evalResult); %#ok<AGROW>
    end
end

function pred = trainPredictMLClassifier(Xtrain, ytrain, Xtest, q, methodName, cfg)
    ytrain = ytrain(:);
    if isempty(Xtest)
        pred = zeros(0, 1);
        return;
    end
    if numel(unique(ytrain)) < 2
        pred = repmat(ytrain(1), size(Xtest, 1), 1);
        return;
    end

    switch string(methodName)
        case "CART"
            model = fitctree(Xtrain, ytrain);
            pred = predict(model, Xtest);
        case "KNN"
            k = min(cfg.knnK, size(Xtrain, 1));
            model = fitcknn(Xtrain, ytrain, 'NumNeighbors', k, 'Standardize', true);
            pred = predict(model, Xtest);
        case "SVM"
            svmTemplate = templateSVM('KernelFunction', 'rbf', 'Standardize', true);
            model = fitcecoc(Xtrain, ytrain, 'Learners', svmTemplate, 'Coding', 'onevsone');
            pred = predict(model, Xtest);
        case "RandomForest"
            model = fitcensemble(Xtrain, ytrain, 'Method', 'Bag', ...
                'NumLearningCycles', cfg.rfNumTrees, 'Learners', 'tree');
            pred = predict(model, Xtest);
        otherwise
            error('Unknown machine learning method: %s', methodName);
    end

    pred = normalizePredictedLabels(pred, q);
end

function rows = runActiveStrategyComparison(Xtrain, ytrain, Xtest, ytest, q, cfg, dataName, run, datasetIndex)
    rows = emptyResultRows();
    strategies = ["AL-Rand", "AL-Entropy", "TWD-AL"];

    for s = 1:numel(strategies)
        strategy = strategies(s);
        rng(cfg.seed + 100000 * datasetIndex + 1000 * run + s, 'twister');
        try
            evalResult = runActiveLexPLStrategy(Xtrain, ytrain, Xtest, ytest, q, cfg, strategy);
        catch ME
            warning('Active strategy %s failed on %s run %d: %s', strategy, dataName, run, ME.message);
            evalResult = makeEvalResult(makeNanMetrics(), 0, 0, 0, "Failed");
        end
        rows(end+1) = makeResultRow(dataName, run, strategy, evalResult); %#ok<AGROW>
    end
end

function evalResult = runActiveLexPLStrategy(Xtrain0, ytrain0, Xpool, ypool, q, cfg, strategy)
    tStart = tic;
    nPool = numel(ypool);
    k = min(cfg.K, nPool);

    if nPool == 0 || k <= 0
        metrics = calculateOrderedMetrics(ypool, nan(size(ypool)), q);
        evalResult = makeEvalResult(metrics, 0, 0, toc(tStart), "EmptyPool");
        return;
    end

    model0 = trainLexPLWeighted(Xtrain0, ytrain0, ones(numel(ytrain0), 1), q, cfg);
    initialPred = predictLexPL(model0, Xpool, cfg);

    switch string(strategy)
        case "AL-Rand"
            selected = randperm(nPool, k)';
            stopReason = "RandomQuery";

        case "AL-Entropy"
            risk = computeTWDInfo(model0, Xtrain0, ytrain0, Xpool, cfg);
            entropyScore = -sum(risk.MS .* log(max(risk.MS, realmin)), 2);
            selected = chooseTopKByScore(entropyScore, (1:nPool)', k);
            stopReason = "EntropyQuery";

        case "TWD-AL"
            risk = computeTWDInfo(model0, Xtrain0, ytrain0, Xpool, cfg);
            negIdx = find(risk.Region == 3);
            selected = chooseTopKByScore(-risk.pS, negIdx, min(k, numel(negIdx)));

            if cfg.twdFillToBudget && numel(selected) < k
                remain = setdiff((1:nPool)', selected, 'stable');
                addIdx = chooseTopKByScore(-risk.pS, remain, k - numel(selected));
                selected = [selected; addIdx];
            end
            selected = selected(1:min(numel(selected), k));
            stopReason = "TWDLowSupportQuery";

        otherwise
            error('Unknown active learning strategy: %s', strategy);
    end

    selected = unique(selected(:), 'stable');
    selected = selected(selected >= 1 & selected <= nPool);

    if isempty(selected)
        corrections = 0;
    else
        corrections = sum(initialPred(selected) ~= ypool(selected));
    end

    XfinalTrain = [Xtrain0; Xpool(selected, :)];
    yfinalTrain = [ytrain0; ypool(selected)];
    finalModel = trainLexPLWeighted(XfinalTrain, yfinalTrain, ones(numel(yfinalTrain), 1), q, cfg);
    finalPred = predictLexPL(finalModel, Xpool, cfg);

    if ~isempty(selected)
        finalPred(selected) = ypool(selected);
    end

    metrics = calculateOrderedMetrics(ypool, finalPred, q);
    evalResult = makeEvalResult(metrics, numel(selected), corrections, toc(tStart), stopReason);
end

function pred = normalizePredictedLabels(pred, q)
    if iscell(pred)
        pred = str2double(pred);
    elseif iscategorical(pred)
        pred = double(pred);
    end
    pred = double(pred(:));
    pred = round(pred);
    pred(pred < 1) = 1;
    pred(pred > q) = q;
end

function metrics = makeNanMetrics()
    metrics.Accuracy = NaN;
    metrics.Precision = NaN;
    metrics.Recall = NaN;
    metrics.Fmeasure = NaN;
    metrics.MAE = NaN;
end

function Summary = summarizeByMethods(RawResults, methodOrder)
    datasets = unique(RawResults.Dataset, 'stable');
    rows = struct('Dataset', {}, 'Method', {}, 'Accuracy', {}, 'Precision', {}, ...
                  'Recall', {}, 'Fmeasure', {}, 'MAE', {}, 'CR', {}, 'Queries', {}, 'Time', {});

    for d = 1:numel(datasets)
        for methodId = 1:numel(methodOrder)
            idx = RawResults.Dataset == datasets(d) & RawResults.Method == methodOrder(methodId);
            if ~any(idx)
                continue;
            end
            r.Dataset = datasets(d);
            r.Method = methodOrder(methodId);
            r.Accuracy  = pm(RawResults.Accuracy(idx), 4);
            r.Precision = pm(RawResults.Precision(idx), 4);
            r.Recall    = pm(RawResults.Recall(idx), 4);
            r.Fmeasure  = pm(RawResults.Fmeasure(idx), 4);
            r.MAE       = pm(RawResults.MAE(idx), 4);
            r.Queries   = pm(RawResults.Queries(idx), 2);
            r.Time      = pm(RawResults.Time(idx), 2);
            if all(RawResults.Queries(idx) == 0)
                r.CR = "-";
            else
                r.CR = pm(RawResults.CorrectionRate(idx), 4);
            end
            rows(end+1) = r; %#ok<AGROW>
        end
    end
    Summary = struct2table(rows);

%% ============================================================
%% TWD risk information and compatible-model sampling
%% ============================================================

function risk = computeTWDInfo(model, Xref, yref, Xpool, cfg)
    if isempty(Xpool)
        risk = struct('pred', [], 'Ucentral', [], 'MS', [], 'samples', [], ...
            'pS', [], 'alpha', [], 'beta', [], 'Region', []);
        return;
    end

    risk = computeRobustInfo(model, Xref, yref, Xpool, cfg);
    MS = risk.MS;
    nPool = size(Xpool, 1);
    rowIdx = (1:nPool)';
    risk.pS = MS(sub2ind(size(MS), rowIdx, risk.pred(:)));

    alpha = ((1 - cfg.xi) * cfg.MAC) / (((1 - cfg.xi) * cfg.MAC) + cfg.xi * (1 - cfg.MAC));
    beta  = (cfg.xi * cfg.MAC) / ((cfg.xi * cfg.MAC) + ((1 - cfg.xi) * (1 - cfg.MAC)));

    Region = 2 * ones(nPool, 1);  % 1=POS, 2=BND, 3=NEG
    Region(risk.pS >= alpha) = 1;
    Region(risk.pS <= beta) = 3;

    risk.alpha = alpha;
    risk.beta = beta;
    risk.Region = Region;
end

function rob = computeRobustInfo(model, Xref, yref, Xpool, cfg)
    q = model.q;
    [pred, Ucentral] = predictMMUTADIS(model, Xpool);
    samples = sampleCompatibleAdditiveModels(Xref, yref, Xpool, q, model.phiStar, cfg);

    rob.pred = pred;
    rob.Ucentral = Ucentral;
    rob.MS = samples.MS;
    rob.samples = samples;
end

function samples = sampleCompatibleAdditiveModels(Xref, yref, Xpool, q, phiStar, cfg)
    lp = buildStage1LP(Xref, yref, q, cfg);

    row = zeros(1, lp.nVar);
    row(lp.idxE) = 1;
    Acomp = [lp.A; sparse(row)];
    rhsComp = [lp.rhs; phiStar + cfg.objectiveSlack];

    solutionSamples = sampleStage1CompatibleModelsRandomObjectiveMix(Acomp, rhsComp, lp, cfg.Ns, cfg);
    nAccepted = size(solutionSamples, 1);
    if nAccepted == 0
        error('Stage-1 compatible-model sampling failed: no feasible model was generated.');
    end

    nPool = size(Xpool, 1);
    predSamples = zeros(nPool, nAccepted);
    USamples = zeros(nPool, nAccepted);
    bSamples = zeros(q - 1, nAccepted);
    counts = zeros(nPool, q);

    for s = 1:nAccepted
        sol = solutionSamples(s, :)';
        thetaS = sol(lp.idxTheta);
        bS = sol(lp.idxB);
        [predS, US] = predictByThetaAndThresholds(thetaS, bS, Xpool, cfg.L, q);

        predSamples(:, s) = predS;
        USamples(:, s) = US;
        bSamples(:, s) = bS(:);

        for i = 1:nPool
            counts(i, predS(i)) = counts(i, predS(i)) + 1;
        end
    end

    samples.predSamples = predSamples;
    samples.USamples = USamples;
    samples.bSamples = bSamples;
    samples.MS = counts ./ nAccepted;
    samples.nAccepted = nAccepted;
end

function sampleMatrix = sampleStage1CompatibleModelsRandomObjectiveMix(Acomp, rhsComp, lp, Ns, cfg)
    poolTarget = cfg.randomObjectivePoolSize;
    mixSize = max(1, cfg.randomObjectiveMixSize);
    maxTrials = max(poolTarget * cfg.randomObjectiveMaxTrialsFactor, poolTarget + 10);
    feasTol = cfg.randomObjectiveFeasTol;

    randomizeE = false;
    if isfield(cfg, 'randomObjectiveRandomizeE') && ~isempty(cfg.randomObjectiveRandomizeE)
        randomizeE = cfg.randomObjectiveRandomizeE;
    end

    eJitter = 0;
    if isfield(cfg, 'randomObjectiveEJitter') && ~isempty(cfg.randomObjectiveEJitter)
        eJitter = cfg.randomObjectiveEJitter;
    end

    vertexPool = zeros(poolTarget, lp.nVar);
    vertexCount = 0;
    trial = 0;

    while vertexCount < poolTarget && trial < maxTrials
        trial = trial + 1;
        f = zeros(lp.nVar, 1);
        f(lp.idxTheta) = randn(numel(lp.idxTheta), 1);
        f(lp.idxB) = randn(numel(lp.idxB), 1);

        if randomizeE
            f(lp.idxE) = randn(numel(lp.idxE), 1);
        elseif eJitter > 0
            f(lp.idxE) = eJitter * randn(numel(lp.idxE), 1);
        end

        try
            sol = runLinprog(f, Acomp, rhsComp, lp.Aeq, lp.beq, lp.lb, lp.ub, cfg);
            if isFeasiblePoint(sol, Acomp, rhsComp, lp.Aeq, lp.beq, lp.lb, lp.ub, 10 * feasTol)
                vertexCount = vertexCount + 1;
                vertexPool(vertexCount, :) = sol(:)';
            end
        catch
            % Individual random-objective LP failures are skipped.
        end
    end

    vertexPool = vertexPool(1:vertexCount, :);
    if isempty(vertexPool)
        error('RandomObjectiveMix failed: no feasible random-objective vertices were generated.');
    end

    if size(vertexPool, 1) > 1
        key = round(vertexPool * 1e10) / 1e10;
        [~, ia] = unique(key, 'rows', 'stable');
        vertexPool = vertexPool(ia, :);
    end

    nVertices = size(vertexPool, 1);
    sampleMatrix = zeros(Ns, lp.nVar);

    for s = 1:Ns
        actualMixSize = min(mixSize, nVertices);
        if nVertices >= actualMixSize
            pick = randperm(nVertices, actualMixSize);
        else
            pick = randi(nVertices, actualMixSize, 1);
        end
        weights = -log(max(rand(actualMixSize, 1), realmin));
        weights = weights ./ sum(weights);
        sampleMatrix(s, :) = weights' * vertexPool(pick, :);
    end
end

function selected = chooseTopKByScore(score, candidates, k)
    candidates = candidates(:);
    if isempty(candidates) || k <= 0
        selected = zeros(0, 1);
        return;
    end

    sub = score(candidates);
    sub(~isfinite(sub)) = -inf;
    if ~any(isfinite(sub))
        selected = candidates(1:min(k, numel(candidates)));
        return;
    end

    [~, ord] = sortrows([-sub(:), candidates(:)]);
    candidates = candidates(ord);
    selected = candidates(1:min(k, numel(candidates)));
end

function model = trainLexPLWeighted(X, y, sampleWeight, q, cfg)
    model = trainWeightedMMUTADIS(X, y, sampleWeight, q, cfg);
    model.type = "Lex-PL";
end

function model = trainWeightedMMUTADIS(X, y, sampleWeight, q, cfg)
    sampleWeight = validateSampleWeights(sampleWeight, numel(y));
    [~, ~, phiStar] = solveUTADISStage1Weighted(X, y, sampleWeight, q, cfg);
    [theta, b, delta] = solveUTADISStage2Weighted(X, y, sampleWeight, q, phiStar, cfg);

    model.type = "Weighted-MM-UTADIS";
    model.q = q;
    model.L = cfg.L;
    model.theta = theta;
    model.b = b(:);
    model.delta = delta;
    model.phiStar = phiStar;
    model.sampleWeight = sampleWeight(:);
end

function [theta, b, phiStar] = solveUTADISStage1Weighted(X, y, sampleWeight, q, cfg)
    lp = buildStage1LP(X, y, q, cfg);
    sampleWeight = validateSampleWeights(sampleWeight, numel(y));

    f = zeros(lp.nVar, 1);
    f(lp.idxE) = sampleWeight(:);
    [sol, fval] = runLinprog(f, lp.A, lp.rhs, lp.Aeq, lp.beq, lp.lb, lp.ub, cfg);

    theta = sol(lp.idxTheta);
    b = sol(lp.idxB);
    phiStar = fval;
end

function [theta, b, delta] = solveUTADISStage2Weighted(X, y, sampleWeight, q, phiStar, cfg)
    sampleWeight = validateSampleWeights(sampleWeight, numel(y));

    n = size(X, 1);
    L = cfg.L;
    Phi = computePiecewiseLinearBasis(X, L);
    p = size(Phi, 2);
    nB = q - 1;

    idxTheta = 1:p;
    idxB = p + (1:nB);
    idxE = p + nB + (1:n);
    idxDelta = p + nB + n + 1;
    nVar = idxDelta;

    A = sparse(0, nVar);
    rhs = [];

    row = zeros(1, nVar);
    row(idxE) = sampleWeight(:)';
    A = [A; sparse(row)];
    rhs = [rhs; phiStar];

    for i = 1:n
        h = y(i);
        if h > 1
            row = zeros(1, nVar);
            row(idxTheta) = -Phi(i, :);
            row(idxB(h - 1)) = 1;
            row(idxDelta) = 1;
            row(idxE(i)) = -1;
            A = [A; sparse(row)]; %#ok<AGROW>
            rhs = [rhs; 0]; %#ok<AGROW>
        end
        if h < q
            row = zeros(1, nVar);
            row(idxTheta) = Phi(i, :);
            row(idxB(h)) = -1;
            row(idxDelta) = 1;
            row(idxE(i)) = -1;
            A = [A; sparse(row)]; %#ok<AGROW>
            rhs = [rhs; 0]; %#ok<AGROW>
        end
    end

    row = zeros(1, nVar);
    row(idxB(1)) = -1;
    row(idxDelta) = 1;
    A = [A; sparse(row)];
    rhs = [rhs; 0];

    row = zeros(1, nVar);
    row(idxB(q - 1)) = 1;
    row(idxDelta) = 1;
    A = [A; sparse(row)];
    rhs = [rhs; 1];

    for h = 2:q - 1
        row = zeros(1, nVar);
        row(idxB(h - 1)) = 1;
        row(idxB(h)) = -1;
        row(idxDelta) = 2;
        A = [A; sparse(row)]; %#ok<AGROW>
        rhs = [rhs; 0]; %#ok<AGROW>
    end

    f = zeros(nVar, 1);
    f(idxDelta) = -1;

    Aeq = sparse(1, idxTheta, ones(1, p), 1, nVar);
    beq = 1;
    lb = zeros(nVar, 1);
    ub = inf(nVar, 1);
    ub(idxB) = 1;
    ub(idxDelta) = 1 / (2 * (q - 1));

    [sol, ~] = runLinprog(f, A, rhs, Aeq, beq, lb, ub, cfg);
    theta = sol(idxTheta);
    b = sol(idxB);
    delta = sol(idxDelta);
end

function lp = buildStage1LP(X, y, q, cfg)
    n = size(X, 1);
    L = cfg.L;
    Phi = computePiecewiseLinearBasis(X, L);
    p = size(Phi, 2);
    nB = q - 1;

    idxTheta = 1:p;
    idxB = p + (1:nB);
    idxE = p + nB + (1:n);
    nVar = p + nB + n;

    A = sparse(0, nVar);
    rhs = [];

    for i = 1:n
        h = y(i);
        if h > 1
            row = zeros(1, nVar);
            row(idxTheta) = -Phi(i, :);
            row(idxB(h - 1)) = 1;
            row(idxE(i)) = -1;
            A = [A; sparse(row)]; %#ok<AGROW>
            rhs = [rhs; 0]; %#ok<AGROW>
        end
        if h < q
            row = zeros(1, nVar);
            row(idxTheta) = Phi(i, :);
            row(idxB(h)) = -1;
            row(idxE(i)) = -1;
            A = [A; sparse(row)]; %#ok<AGROW>
            rhs = [rhs; 0]; %#ok<AGROW>
        end
    end

    [A, rhs] = addThresholdOrderRows(A, rhs, idxB, q, cfg.thresholdMinGap, nVar);

    Aeq = sparse(1, idxTheta, ones(1, p), 1, nVar);
    beq = 1;
    lb = zeros(nVar, 1);
    ub = inf(nVar, 1);
    ub(idxB) = 1;

    lp.A = A;
    lp.rhs = rhs;
    lp.Aeq = Aeq;
    lp.beq = beq;
    lp.lb = lb;
    lp.ub = ub;
    lp.idxTheta = idxTheta;
    lp.idxB = idxB;
    lp.idxE = idxE;
    lp.nVar = nVar;
end

function [A, rhs] = addThresholdOrderRows(A, rhs, idxB, q, minGap, nVar)
    if q <= 1
        return;
    end

    row = zeros(1, nVar);
    row(idxB(1)) = -1;
    A = [A; sparse(row)];
    rhs = [rhs; -minGap];

    for h = 2:q - 1
        row = zeros(1, nVar);
        row(idxB(h - 1)) = 1;
        row(idxB(h)) = -1;
        A = [A; sparse(row)]; %#ok<AGROW>
        rhs = [rhs; -minGap]; %#ok<AGROW>
    end

    row = zeros(1, nVar);
    row(idxB(q - 1)) = 1;
    A = [A; sparse(row)];
    rhs = [rhs; 1 - minGap];
end

function [pred, U] = predictLexPL(model, X, cfg) %#ok<INUSD>
    [pred, U] = predictMMUTADIS(model, X);
end

function [pred, U] = predictMMUTADIS(model, X)
    [pred, U] = predictByThetaAndThresholds(model.theta, model.b, X, model.L, model.q);
end

function [prediction, utility] = predictByThetaAndThresholds(theta, thresholds, X, numSegments, numClasses)
    utility = computePiecewiseLinearBasis(X, numSegments) * theta;
    prediction = assignByThresholds(utility, thresholds, numClasses);
end

function prediction = assignByThresholds(utility, thresholds, numClasses)
    prediction = ones(size(utility));
    for classId = 1:(numClasses - 1)
        prediction = prediction + (utility > thresholds(classId) + 1e-10);
    end
    prediction = max(1, min(numClasses, prediction));
end

function Phi = computePiecewiseLinearBasis(X, numSegments)
    [numSamples, numCriteria] = size(X);
    breakpoints = linspace(0, 1, numSegments + 1);
    Phi = zeros(numSamples, numCriteria * numSegments);

    col = 0;
    for criterionId = 1:numCriteria
        x = X(:, criterionId);
        for segmentId = 1:numSegments
            col = col + 1;
            segmentWidth = breakpoints(segmentId + 1) - breakpoints(segmentId);
            Phi(:, col) = max(0, min(1, (x - breakpoints(segmentId)) ./ segmentWidth));
        end
    end
end

function w = validateSampleWeights(w, n)
    if nargin < 1 || isempty(w)
        w = ones(n, 1);
    end
    w = double(w(:));
    if numel(w) ~= n
        error('Sample weight length mismatch: numel(w)=%d, n=%d.', numel(w), n);
    end
    if any(~isfinite(w)) || any(w <= 0)
        error('Sample weights must be positive and finite.');
    end
end

%% ============================================================
%% Data loading, splitting, metrics, summary
%% ============================================================

function [X, y, alternativeNames, criterionNames] = readMCSDataset(filePath)
    if ~exist(filePath, 'file')
        error('Cannot find data file: %s', filePath);
    end

    try
        dataTable = readtable(filePath, 'VariableNamingRule', 'preserve');
    catch
        dataTable = readtable(filePath);
    end

    variableNames = string(dataTable.Properties.VariableNames);
    classCol = find(strcmpi(variableNames, 'Class'), 1);
    alternativeCol = find(strcmpi(variableNames, 'Alternative'), 1);
    criterionCols = find(startsWith(variableNames, 'g', 'IgnoreCase', true));

    if isempty(classCol)
        error('The data file must contain a Class column.');
    end
    if isempty(criterionCols)
        error('The data file must contain criterion columns named g1, g2, ...');
    end

    X = double(table2array(dataTable(:, criterionCols)));
    X = normalizeToUnitIntervalIfNeeded(X);
    y = parseOrderedClassLabels(dataTable{:, classCol});

    if isempty(alternativeCol)
        alternativeNames = "a" + string((1:size(X, 1))');
    else
        alternativeNames = string(dataTable{:, alternativeCol});
    end

    criterionNames = variableNames(criterionCols);
end

function X = normalizeToUnitIntervalIfNeeded(X)
    for criterionId = 1:size(X, 2)
        xmin = min(X(:, criterionId));
        xmax = max(X(:, criterionId));

        if xmin < -1e-10 || xmax > 1 + 1e-10
            if xmax > xmin
                X(:, criterionId) = (X(:, criterionId) - xmin) ./ (xmax - xmin);
            else
                X(:, criterionId) = 0;
            end
        end
    end
end

function y = parseOrderedClassLabels(rawLabels)
    rawLabels = string(rawLabels);
    y = zeros(numel(rawLabels), 1);

    for sampleId = 1:numel(rawLabels)
        token = regexp(rawLabels(sampleId), '\d+', 'match');
        if ~isempty(token)
            y(sampleId) = str2double(token{1});
        else
            numericValue = str2double(rawLabels(sampleId));
            if isnan(numericValue)
                error('Class labels must be numeric or formatted as Cl1, Cl2, ...');
            end
            y(sampleId) = numericValue;
        end
    end
end

function [trainIdx, testIdx] = splitReferenceAndPool(y, cfg)
    n = numel(y);
    idx = randperm(n)';
    nTrain = round(cfg.trainRatio * n);
    if n >= 2
        nTrain = max(1, min(nTrain, n - 1));
    else
        nTrain = 1;
    end
    trainIdx = idx(1:nTrain);
    testIdx = idx(nTrain + 1:end);
end

function metrics = calculateOrderedMetrics(yTrue, yPred, numClasses)
    yTrue = yTrue(:);
    yPred = yPred(:);
    validMask = ~isnan(yPred);
    yTrue = yTrue(validMask);
    yPred = yPred(validMask);

    if isempty(yTrue)
        metrics.Accuracy = NaN;
        metrics.Precision = NaN;
        metrics.Recall = NaN;
        metrics.Fmeasure = NaN;
        metrics.MAE = NaN;
        return;
    end

    confusionMatrix = zeros(numClasses, numClasses);
    for idx = 1:numel(yTrue)
        if yTrue(idx) >= 1 && yTrue(idx) <= numClasses && yPred(idx) >= 1 && yPred(idx) <= numClasses
            confusionMatrix(yTrue(idx), yPred(idx)) = confusionMatrix(yTrue(idx), yPred(idx)) + 1;
        end
    end

    precision = nan(numClasses, 1);
    recall = nan(numClasses, 1);
    fmeasure = nan(numClasses, 1);

    for classId = 1:numClasses
        truePositive = confusionMatrix(classId, classId);
        falsePositive = sum(confusionMatrix(:, classId)) - truePositive;
        falseNegative = sum(confusionMatrix(classId, :)) - truePositive;

        if truePositive + falsePositive > 0
            precision(classId) = truePositive / (truePositive + falsePositive);
        end
        if truePositive + falseNegative > 0
            recall(classId) = truePositive / (truePositive + falseNegative);
        end
        if ~isnan(precision(classId)) && ~isnan(recall(classId)) && precision(classId) + recall(classId) > 0
            fmeasure(classId) = 2 * precision(classId) * recall(classId) / (precision(classId) + recall(classId));
        end
    end

    metrics.Accuracy = mean(yPred == yTrue);
    metrics.Precision = mean(precision, 'omitnan');
    metrics.Recall = mean(recall, 'omitnan');
    metrics.Fmeasure = mean(fmeasure, 'omitnan');
    metrics.MAE = mean(abs(yPred - yTrue));
end

function evalResult = makeEvalResult(metrics, queries, corrections, timeUsed, stopReason)
    evalResult.Accuracy = metrics.Accuracy;
    evalResult.Precision = metrics.Precision;
    evalResult.Recall = metrics.Recall;
    evalResult.Fmeasure = metrics.Fmeasure;
    evalResult.MAE = metrics.MAE;
    evalResult.Queries = queries;
    evalResult.Corrections = corrections;
    if queries > 0
        evalResult.CorrectionRate = corrections / queries;
    else
        evalResult.CorrectionRate = NaN;
    end
    evalResult.Time = timeUsed;
    evalResult.StopReason = string(stopReason);
end

function row = makeResultRow(datasetName, run, methodName, evalResult)
    row.Dataset = string(datasetName);
    row.Run = run;
    row.Method = string(methodName);
    row.Accuracy = evalResult.Accuracy;
    row.Precision = evalResult.Precision;
    row.Recall = evalResult.Recall;
    row.Fmeasure = evalResult.Fmeasure;
    row.MAE = evalResult.MAE;
    row.Queries = evalResult.Queries;
    row.Corrections = evalResult.Corrections;
    row.CorrectionRate = evalResult.CorrectionRate;
    row.Time = evalResult.Time;
    row.StopReason = string(evalResult.StopReason);
end

function rows = emptyResultRows()
    rows = struct('Dataset', {}, 'Run', {}, 'Method', {}, 'Accuracy', {}, ...
        'Precision', {}, 'Recall', {}, 'Fmeasure', {}, 'MAE', {}, 'Queries', {}, ...
        'Corrections', {}, 'CorrectionRate', {}, 'Time', {}, 'StopReason', {});
end

function s = pm(x, digits)
    x = x(~isnan(x));
    if isempty(x)
        s = "-";
        return;
    end
    mu = mean(x);
    sigma = std(x);
    fmt = "%0." + string(digits) + "f ± %0." + string(digits) + "f";
    s = string(sprintf(fmt, mu, sigma));
end

%% ============================================================
%% Configuration, path, and solver helpers
%% ============================================================

function filePath = resolveDataFilePath(dataDir, fileName)
    candidatePaths = [
        string(fullfile(dataDir, fileName))
        string(fullfile(pwd, fileName))
        string(fullfile(pwd, 'data', fileName))
        string(fullfile('/home/shenyufeng/Documents/MATLAB/Datasets', fileName))
        string(fullfile('/home/shenyufeng/Documents/MATLAB/Code/download-master/monotone-classification-problems', fileName))
    ];
    scriptPath = mfilename('fullpath');
    if ~isempty(scriptPath)
        scriptDir = fileparts(scriptPath);
        candidatePaths(end+1, 1) = string(fullfile(scriptDir, fileName));
        candidatePaths(end+1, 1) = string(fullfile(scriptDir, 'data', fileName));
        candidatePaths(end+1, 1) = string(fullfile(scriptDir, 'monotone-classification-problems', fileName));
    end
    for pathId = 1:numel(candidatePaths)
        if exist(char(candidatePaths(pathId)), 'file')
            filePath = char(candidatePaths(pathId));
            return;
        end
    end
    error('找不到数据文件：%s。', fileName);
end

function validateComparisonConfig(cfg)
    if cfg.trainRatio <= 0 || cfg.trainRatio >= 1
        error('cfg.trainRatio must be between 0 and 1.');
    end
    if cfg.nRuns < 1 || fix(cfg.nRuns) ~= cfg.nRuns
        error('cfg.nRuns must be a positive integer.');
    end
    if cfg.K < 1 || fix(cfg.K) ~= cfg.K
        error('cfg.K must be a positive integer.');
    end
    if cfg.Ns < 1 || fix(cfg.Ns) ~= cfg.Ns
        error('cfg.Ns must be a positive integer.');
    end
    if cfg.objectiveSlack < 0
        error('cfg.objectiveSlack must be nonnegative.');
    end
    if cfg.randomObjectivePoolSize < 1 || fix(cfg.randomObjectivePoolSize) ~= cfg.randomObjectivePoolSize
        error('cfg.randomObjectivePoolSize must be a positive integer.');
    end
    if cfg.randomObjectiveMixSize < 1 || fix(cfg.randomObjectiveMixSize) ~= cfg.randomObjectiveMixSize
        error('cfg.randomObjectiveMixSize must be a positive integer.');
    end
end

function validateDatasetForComparison(X, y, q, fileName)
    if isempty(X) || isempty(y)
        error('Dataset %s is empty.', fileName);
    end
    if size(X, 1) ~= numel(y)
        error('Dataset %s has inconsistent feature and label counts.', fileName);
    end
    if any(isnan(X(:))) || any(isnan(y(:)))
        error('Dataset %s contains NaN values.', fileName);
    end
    if any(y < 1) || any(y > q) || any(fix(y) ~= y)
        error('Dataset %s contains invalid ordered class labels.', fileName);
    end
end

function [x, fval] = runLinprog(f, A, b, Aeq, beq, lb, ub, cfg)
    if isfield(cfg, 'linprogAlgorithms') && ~isempty(cfg.linprogAlgorithms)
        algList = string(cfg.linprogAlgorithms);
    else
        algList = ["dual-simplex", "interior-point", "default"];
    end

    if isfield(cfg, 'linprogRelaxedTolerances') && ~isempty(cfg.linprogRelaxedTolerances)
        tolList = cfg.linprogRelaxedTolerances(:)';
    else
        tolList = [NaN, 1e-5];
    end

    if isfield(cfg, 'solverDisplay') && ~isempty(cfg.solverDisplay)
        solverDisplay = cfg.solverDisplay;
    else
        solverDisplay = 'none';
    end

    reports = strings(0, 1);
    lastME = [];

    for tt = 1:numel(tolList)
        relaxedTol = tolList(tt);
        for aa = 1:numel(algList)
            alg = algList(aa);
            try
                opts = makeLinprogOptions(solverDisplay, alg);
                if ~isnan(relaxedTol)
                    opts = trySetLinprogOption(opts, 'ConstraintTolerance', relaxedTol);
                    opts = trySetLinprogOption(opts, 'OptimalityTolerance', relaxedTol);
                end

                [x, fval, exitflag, output] = linprog(f, A, b, Aeq, beq, lb, ub, opts);
                if exitflag > 0
                    if isfield(cfg, 'linprogVerboseFallback') && cfg.linprogVerboseFallback && (aa > 1 || tt > 1)
                        fprintf('linprog recovered by alg=%s, tol=%g.\n', char(alg), relaxedTol);
                    end
                    return;
                end

                msg = "";
                if exist('output', 'var') && isfield(output, 'message')
                    msg = string(output.message);
                end
                reports(end+1, 1) = sprintf('linprog failed: alg=%s, tol=%g, exitflag=%d, message=%s', ...
                    char(alg), relaxedTol, exitflag, char(msg)); %#ok<AGROW>
            catch ME
                lastME = ME; %#ok<NASGU>
                reports(end+1, 1) = sprintf('linprog crashed: alg=%s, tol=%g, identifier=%s, message=%s', ...
                    char(alg), relaxedTol, ME.identifier, ME.message); %#ok<AGROW>
            end
        end
    end

    failFile = '';
    if ~isfield(cfg, 'saveLinprogFailure') || cfg.saveLinprogFailure
        try
            if isfield(cfg, 'linprogFailureDir') && ~isempty(cfg.linprogFailureDir)
                failDir = cfg.linprogFailureDir;
            else
                failDir = pwd;
            end
            if ~exist(failDir, 'dir')
                mkdir(failDir);
            end
            failFile = [tempname(failDir), '.mat'];
            save(failFile, 'f', 'A', 'b', 'Aeq', 'beq', 'lb', 'ub', 'cfg', 'reports', 'lastME');
        catch
        end
    end

    if ~isempty(failFile)
        error('linprog failed; failure LP saved to: %s\n%s', failFile, char(strjoin(reports, newline)));
    else
        error('linprog failed; all algorithms/tolerances failed:\n%s', char(strjoin(reports, newline)));
    end
end

function opts = makeLinprogOptions(solverDisplay, alg)
    if string(alg) == "default"
        opts = optimoptions('linprog', 'Display', solverDisplay);
    else
        opts = optimoptions('linprog', 'Display', solverDisplay, 'Algorithm', char(alg));
    end
end

function opts = trySetLinprogOption(opts, name, value)
    try
        opts = optimoptions(opts, name, value);
    catch
        try
            opts.(name) = value;
        catch
        end
    end
end

function ok = isFeasiblePoint(x, A, rhs, Aeq, beq, lb, ub, tol)
    ineqViolation = max(A * x - rhs);
    if isempty(ineqViolation)
        ineqViolation = -inf;
    end

    eqViolation = max(abs(Aeq * x - beq));
