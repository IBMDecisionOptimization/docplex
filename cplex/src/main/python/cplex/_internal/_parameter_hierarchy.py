# --------------------------------------------------------------------------
# Version 22.2.0
# --------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2000, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# --------------------------------------------------------------------------
# pylint: disable=line-too-long
"""Parameter hierarchy for the CPLEX Python API.

"""
from . import _parameters_auto as _auto
from . import _parameter_classes as _cls


def barrier_limits_members(env, parent):
    """Limits for barrier optimization."""
    return dict(
        _name="limits",
        help=lambda: "Limits for barrier optimization.",
        corrections=_cls.NumParameter(env, _auto.BarrierLimitsCorrections, parent, 'corrections'),
        growth=_cls.NumParameter(env, _auto.BarrierLimitsGrowth, parent, 'growth'),
        iteration=_cls.NumParameter(env, _auto.BarrierLimitsIteration, parent, 'iteration'),
        objrange=_cls.NumParameter(env, _auto.BarrierLimitsObjRange, parent, 'objrange'),
    )

def barrier_members(env, parent):
    """Parameters for barrier optimization."""
    return dict(
        _name="barrier",
        help=lambda: "Parameters for barrier optimization.",
        algorithm=_cls.NumParameter(env, _auto.BarrierAlgorithm, parent, 'algorithm', _cls.bar_alg_constants),
        colnonzeros=_cls.NumParameter(env, _auto.BarrierColNonzeros, parent, 'colnonzeros'),
        convergetol=_cls.NumParameter(env, _auto.BarrierConvergeTol, parent, 'convergetol'),
        crossover=_cls.NumParameter(env, _auto.BarrierCrossover, parent, 'crossover', _cls.crossover_constants),
        display=_cls.NumParameter(env, _auto.BarrierDisplay, parent, 'display', _cls.display_constants),
        limits=_cls.ParameterGroup(env, barrier_limits_members, parent),
        ordering=_cls.NumParameter(env, _auto.BarrierOrdering, parent, 'ordering', _cls.bar_order_constants),
        qcpconvergetol=_cls.NumParameter(env, _auto.BarrierQCPConvergeTol, parent, 'qcpconvergetol'),
        startalg=_cls.NumParameter(env, _auto.BarrierStartAlg, parent, 'startalg', _cls.bar_start_alg_constants),
    )

def benders_tolerances_members(env, parent):
    """Numerical tolerances for Benders cuts."""
    return dict(
        _name="tolerances",
        help=lambda: "Numerical tolerances for Benders cuts.",
        feasibilitycut=_cls.NumParameter(env, _auto.BendersTolerancesFeasibilityCut, parent, 'feasibilitycut'),
        optimalitycut=_cls.NumParameter(env, _auto.BendersTolerancesOptimalityCut, parent, 'optimalitycut'),
    )

def benders_members(env, parent):
    """Parameters for benders optimization."""
    return dict(
        _name="benders",
        help=lambda: "Parameters for benders optimization.",
        strategy=_cls.NumParameter(env, _auto.BendersStrategy, parent, 'strategy', _cls.benders_strategy_constants),
        tolerances=_cls.ParameterGroup(env, benders_tolerances_members, parent),
        workeralgorithm=_cls.NumParameter(env, _auto.BendersWorkerAlgorithm, parent, 'workeralgorithm', _cls.subalg_constants),
    )

def conflict_members(env, parent):
    """Parameters for finding conflicts."""
    return dict(
        _name="conflict",
        help=lambda: "Parameters for finding conflicts.",
        algorithm=_cls.NumParameter(env, _auto.ConflictAlgorithm, parent, 'algorithm', _cls.conflict_algorithm_constants),
        display=_cls.NumParameter(env, _auto.ConflictDisplay, parent, 'display', _cls.display_constants),
    )

def emphasis_members(env, parent):
    """Optimization emphasis."""
    return dict(
        _name="emphasis",
        help=lambda: "Optimization emphasis.",
        memory=_cls.NumParameter(env, _auto.EmphasisMemory, parent, 'memory', _cls.off_on_constants),
        mip=_cls.NumParameter(env, _auto.EmphasisMIP, parent, 'mip', _cls.mip_emph_constants),
        numerical=_cls.NumParameter(env, _auto.EmphasisNumerical, parent, 'numerical', _cls.off_on_constants),
    )

def feasopt_members(env, parent):
    """Parameters for feasopt."""
    return dict(
        _name="feasopt",
        help=lambda: "Parameters for feasopt.",
        mode=_cls.NumParameter(env, _auto.FeasoptMode, parent, 'mode', _cls.feasopt_mode_constants),
        tolerance=_cls.NumParameter(env, _auto.FeasoptTolerance, parent, 'tolerance'),
    )

def mip_cuts_members(env, parent):
    """Types of cuts used during mixed integer optimization."""
    return dict(
        _name="cuts",
        help=lambda: "Types of cuts used during mixed integer optimization.",
        bqp=_cls.NumParameter(env, _auto.MIPCutsBQP, parent, 'bqp', _cls.v_agg_constants),
        cliques=_cls.NumParameter(env, _auto.MIPCutsCliques, parent, 'cliques', _cls.v_agg_constants),
        covers=_cls.NumParameter(env, _auto.MIPCutsCovers, parent, 'covers', _cls.v_agg_constants),
        disjunctive=_cls.NumParameter(env, _auto.MIPCutsDisjunctive, parent, 'disjunctive', _cls.v_agg_constants),
        flowcovers=_cls.NumParameter(env, _auto.MIPCutsFlowCovers, parent, 'flowcovers', _cls.agg_constants),
        gomory=_cls.NumParameter(env, _auto.MIPCutsGomory, parent, 'gomory', _cls.agg_constants),
        gubcovers=_cls.NumParameter(env, _auto.MIPCutsGUBCovers, parent, 'gubcovers', _cls.agg_constants),
        implied=_cls.NumParameter(env, _auto.MIPCutsImplied, parent, 'implied', _cls.agg_constants),
        liftproj=_cls.NumParameter(env, _auto.MIPCutsLiftProj, parent, 'liftproj', _cls.v_agg_constants),
        localimplied=_cls.NumParameter(env, _auto.MIPCutsLocalImplied, parent, 'localimplied', _cls.v_agg_constants),
        mcfcut=_cls.NumParameter(env, _auto.MIPCutsMCFCut, parent, 'mcfcut', _cls.agg_constants),
        mircut=_cls.NumParameter(env, _auto.MIPCutsMIRCut, parent, 'mircut', _cls.agg_constants),
        nodecuts=_cls.NumParameter(env, _auto.MIPCutsNodecuts, parent, 'nodecuts', _cls.v_agg_constants),
        pathcut=_cls.NumParameter(env, _auto.MIPCutsPathCut, parent, 'pathcut', _cls.agg_constants),
        rlt=_cls.NumParameter(env, _auto.MIPCutsRLT, parent, 'rlt', _cls.v_agg_constants),
        zerohalfcut=_cls.NumParameter(env, _auto.MIPCutsZeroHalfCut, parent, 'zerohalfcut', _cls.agg_constants),
    )

def mip_limits_members(env, parent):
    """Limits for mixed integer optimization."""
    return dict(
        _name="limits",
        help=lambda: "Limits for mixed integer optimization.",
        aggforcut=_cls.NumParameter(env, _auto.MIPLimitsAggForCut, parent, 'aggforcut'),
        auxrootthreads=_cls.NumParameter(env, _auto.MIPLimitsAuxRootThreads, parent, 'auxrootthreads'),
        cutpasses=_cls.NumParameter(env, _auto.MIPLimitsCutPasses, parent, 'cutpasses'),
        cutsfactor=_cls.NumParameter(env, _auto.MIPLimitsCutsFactor, parent, 'cutsfactor'),
        eachcutlimit=_cls.NumParameter(env, _auto.MIPLimitsEachCutLimit, parent, 'eachcutlimit'),
        gomorycand=_cls.NumParameter(env, _auto.MIPLimitsGomoryCand, parent, 'gomorycand'),
        gomorypass=_cls.NumParameter(env, _auto.MIPLimitsGomoryPass, parent, 'gomorypass'),
        lowerobjstop=_cls.NumParameter(env, _auto.MIPLimitsLowerObjStop, parent, 'lowerobjstop'),
        nodes=_cls.NumParameter(env, _auto.MIPLimitsNodes, parent, 'nodes'),
        populate=_cls.NumParameter(env, _auto.MIPLimitsPopulate, parent, 'populate'),
        probedettime=_cls.NumParameter(env, _auto.MIPLimitsProbeDetTime, parent, 'probedettime'),
        probetime=_cls.NumParameter(env, _auto.MIPLimitsProbeTime, parent, 'probetime'),
        repairtries=_cls.NumParameter(env, _auto.MIPLimitsRepairTries, parent, 'repairtries'),
        solutions=_cls.NumParameter(env, _auto.MIPLimitsSolutions, parent, 'solutions'),
        strongcand=_cls.NumParameter(env, _auto.MIPLimitsStrongCand, parent, 'strongcand'),
        strongit=_cls.NumParameter(env, _auto.MIPLimitsStrongIt, parent, 'strongit'),
        treememory=_cls.NumParameter(env, _auto.MIPLimitsTreeMemory, parent, 'treememory'),
        upperobjstop=_cls.NumParameter(env, _auto.MIPLimitsUpperObjStop, parent, 'upperobjstop'),
    )

def mip_polishafter_members(env, parent):
    """Starting conditions for solution polishing."""
    return dict(
        _name="polishafter",
        help=lambda: "Starting conditions for solution polishing.",
        absmipgap=_cls.NumParameter(env, _auto.MIPPolishAfterAbsMIPGap, parent, 'absmipgap'),
        dettime=_cls.NumParameter(env, _auto.MIPPolishAfterDetTime, parent, 'dettime'),
        mipgap=_cls.NumParameter(env, _auto.MIPPolishAfterMIPGap, parent, 'mipgap'),
        nodes=_cls.NumParameter(env, _auto.MIPPolishAfterNodes, parent, 'nodes'),
        solutions=_cls.NumParameter(env, _auto.MIPPolishAfterSolutions, parent, 'solutions'),
        time=_cls.NumParameter(env, _auto.MIPPolishAfterTime, parent, 'time'),
    )

def mip_pool_members(env, parent):
    """Solution pool characteristics."""
    return dict(
        _name="pool",
        help=lambda: "Solution pool characteristics.",
        absgap=_cls.NumParameter(env, _auto.MIPPoolAbsGap, parent, 'absgap'),
        capacity=_cls.NumParameter(env, _auto.MIPPoolCapacity, parent, 'capacity'),
        intensity=_cls.NumParameter(env, _auto.MIPPoolIntensity, parent, 'intensity', _cls.v_agg_constants),
        relgap=_cls.NumParameter(env, _auto.MIPPoolRelGap, parent, 'relgap'),
        replace=_cls.NumParameter(env, _auto.MIPPoolReplace, parent, 'replace', _cls.replace_constants),
    )

def mip_strategy_members(env, parent):
    """Strategy for mixed integer optimization."""
    return dict(
        _name="strategy",
        help=lambda: "Strategy for mixed integer optimization.",
        backtrack=_cls.NumParameter(env, _auto.MIPStrategyBacktrack, parent, 'backtrack'),
        bbinterval=_cls.NumParameter(env, _auto.MIPStrategyBBInterval, parent, 'bbinterval'),
        branch=_cls.NumParameter(env, _auto.MIPStrategyBranch, parent, 'branch', _cls.brdir_constants),
        cardls=_cls.NumParameter(env, _auto.MIPStrategyCardLs, parent, 'cardls', _cls.cardls_constants),
        dive=_cls.NumParameter(env, _auto.MIPStrategyDive, parent, 'dive', _cls.dive_constants),
        file=_cls.NumParameter(env, _auto.MIPStrategyFile, parent, 'file', _cls.file_constants),
        fpheur=_cls.NumParameter(env, _auto.MIPStrategyFPHeur, parent, 'fpheur', _cls.fpheur_constants),
        heuristiceffort=_cls.NumParameter(env, _auto.MIPStrategyHeuristicEffort, parent, 'heuristiceffort'),
        heuristicfreq=_cls.NumParameter(env, _auto.MIPStrategyHeuristicFreq, parent, 'heuristicfreq'),
        kappastats=_cls.NumParameter(env, _auto.MIPStrategyKappaStats, parent, 'kappastats', _cls.kappastats_constants),
        lbheur=_cls.NumParameter(env, _auto.MIPStrategyLBHeur, parent, 'lbheur', _cls.off_on_constants),
        miqcpstrat=_cls.NumParameter(env, _auto.MIPStrategyMIQCPStrat, parent, 'miqcpstrat', _cls.miqcp_constants),
        nodeselect=_cls.NumParameter(env, _auto.MIPStrategyNodeSelect, parent, 'nodeselect', _cls.nodesel_constants),
        order=_cls.NumParameter(env, _auto.MIPStrategyOrder, parent, 'order', _cls.off_on_constants),
        presolvenode=_cls.NumParameter(env, _auto.MIPStrategyPresolveNode, parent, 'presolvenode', _cls.presolve_constants),
        probe=_cls.NumParameter(env, _auto.MIPStrategyProbe, parent, 'probe', _cls.v_agg_constants),
        rinsheur=_cls.NumParameter(env, _auto.MIPStrategyRINSHeur, parent, 'rinsheur'),
        search=_cls.NumParameter(env, _auto.MIPStrategySearch, parent, 'search', _cls.search_constants),
        startalgorithm=_cls.NumParameter(env, _auto.MIPStrategyStartAlgorithm, parent, 'startalgorithm', _cls.alg_constants),
        subalgorithm=_cls.NumParameter(env, _auto.MIPStrategySubAlgorithm, parent, 'subalgorithm', _cls.subalg_constants),
        variableselect=_cls.NumParameter(env, _auto.MIPStrategyVariableSelect, parent, 'variableselect', _cls.varsel_constants),
    )

def mip_submip_members(env, parent):
    """Parameters used when solving sub-MIPs."""
    return dict(
        _name="submip",
        help=lambda: "Parameters used when solving sub-MIPs.",
        startalg=_cls.NumParameter(env, _auto.MIPSubMIPStartAlg, parent, 'startalg', _cls.subalg_constants),
        subalg=_cls.NumParameter(env, _auto.MIPSubMIPSubAlg, parent, 'subalg', _cls.subalg_constants),
        nodelimit=_cls.NumParameter(env, _auto.MIPSubMIPNodeLimit, parent, 'nodelimit'),
        scale=_cls.NumParameter(env, _auto.MIPSubMIPScale, parent, 'scale', _cls.scale_constants),
    )

def mip_tolerances_members(env, parent):
    """Tolerances for mixed integer optimization."""
    return dict(
        _name="tolerances",
        help=lambda: "Tolerances for mixed integer optimization.",
        absmipgap=_cls.NumParameter(env, _auto.MIPTolerancesAbsMIPGap, parent, 'absmipgap'),
        linearization=_cls.NumParameter(env, _auto.MIPTolerancesLinearization, parent, 'linearization'),
        integrality=_cls.NumParameter(env, _auto.MIPTolerancesIntegrality, parent, 'integrality'),
        lowercutoff=_cls.NumParameter(env, _auto.MIPTolerancesLowerCutoff, parent, 'lowercutoff'),
        mipgap=_cls.NumParameter(env, _auto.MIPTolerancesMIPGap, parent, 'mipgap'),
        objdifference=_cls.NumParameter(env, _auto.MIPTolerancesObjDifference, parent, 'objdifference'),
        relobjdifference=_cls.NumParameter(env, _auto.MIPTolerancesRelObjDifference, parent, 'relobjdifference'),
        uppercutoff=_cls.NumParameter(env, _auto.MIPTolerancesUpperCutoff, parent, 'uppercutoff'),
    )

def mip_members(env, parent):
    """Parameters for mixed integer optimization."""
    return dict(
        _name="mip",
        help=lambda: "Parameters for mixed integer optimization.",
        cuts=_cls.ParameterGroup(env, mip_cuts_members, parent),
        display=_cls.NumParameter(env, _auto.MIPDisplay, parent, 'display', _cls.mip_display_constants),
        interval=_cls.NumParameter(env, _auto.MIPInterval, parent, 'interval'),
        limits=_cls.ParameterGroup(env, mip_limits_members, parent),
        ordertype=_cls.NumParameter(env, _auto.MIPOrderType, parent, 'ordertype', _cls.ordertype_constants),
        polishafter=_cls.ParameterGroup(env, mip_polishafter_members, parent),
        pool=_cls.ParameterGroup(env, mip_pool_members, parent),
        strategy=_cls.ParameterGroup(env, mip_strategy_members, parent),
        submip=_cls.ParameterGroup(env, mip_submip_members, parent),
        tolerances=_cls.ParameterGroup(env, mip_tolerances_members, parent),
    )

def multiobjective_members(env, parent):
    """Parameters for multi-objective optimization."""
    return dict(
        _name="multiobjective",
        help=lambda: "Parameters for multi-objective optimization.",
        display=_cls.NumParameter(env, _auto.MultiObjectiveDisplay, parent, 'display', _cls.display_constants),
    )

def network_tolerances_members(env, parent):
    """Numerical tolerances for network simplex optimization."""
    return dict(
        _name="tolerances",
        help=lambda: "Numerical tolerances for network simplex optimization.",
        feasibility=_cls.NumParameter(env, _auto.NetworkTolerancesFeasibility, parent, 'feasibility'),
        optimality=_cls.NumParameter(env, _auto.NetworkTolerancesOptimality, parent, 'optimality'),
    )

def network_members(env, parent):
    """Parameters for network optimizations."""
    return dict(
        _name="network",
        help=lambda: "Parameters for network optimizations.",
        display=_cls.NumParameter(env, _auto.NetworkDisplay, parent, 'display', _cls.network_display_constants),
        iterations=_cls.NumParameter(env, _auto.NetworkIterations, parent, 'iterations'),
        netfind=_cls.NumParameter(env, _auto.NetworkNetFind, parent, 'netfind', _cls.network_netfind_constants),
        pricing=_cls.NumParameter(env, _auto.NetworkPricing, parent, 'pricing', _cls.network_pricing_constants),
        tolerances=_cls.ParameterGroup(env, network_tolerances_members, parent),
    )

def output_members(env, parent):
    """Extent and destinations of outputs."""
    return dict(
        _name="output",
        help=lambda: "Extent and destinations of outputs.",
        clonelog=_cls.NumParameter(env, _auto.OutputCloneLog, parent, 'clonelog', _cls.off_on_constants),
        intsolfileprefix=_cls.StrParameter(env, _auto.OutputIntSolFilePrefix, parent, 'intsolfileprefix'),
        mpslong=_cls.NumParameter(env, _auto.OutputMPSLong, parent, 'mpslong', _cls.off_on_constants),
        writelevel=_cls.NumParameter(env, _auto.OutputWriteLevel, parent, 'writelevel', _cls.writelevel_constants),
    )

def preprocessing_members(env, parent):
    """Parameters for preprocessing."""
    return dict(
        _name="preprocessing",
        help=lambda: "Parameters for preprocessing.",
        aggregator=_cls.NumParameter(env, _auto.PreprocessingAggregator, parent, 'aggregator'),
        boundstrength=_cls.NumParameter(env, _auto.PreprocessingBoundStrength, parent, 'boundstrength', _cls.auto_off_on_constants),
        coeffreduce=_cls.NumParameter(env, _auto.PreprocessingCoeffReduce, parent, 'coeffreduce', _cls.coeffreduce_constants),
        dependency=_cls.NumParameter(env, _auto.PreprocessingDependency, parent, 'dependency', _cls.dependency_constants),
        dual=_cls.NumParameter(env, _auto.PreprocessingDual, parent, 'dual', _cls.dual_constants),
        fill=_cls.NumParameter(env, _auto.PreprocessingFill, parent, 'fill'),
        folding=_cls.NumParameter(env, _auto.PreprocessingFolding, parent, 'folding'),
        linear=_cls.NumParameter(env, _auto.PreprocessingLinear, parent, 'linear', _cls.linear_constants),
        numpass=_cls.NumParameter(env, _auto.PreprocessingNumPass, parent, 'numpass'),
        presolve=_cls.NumParameter(env, _auto.PreprocessingPresolve, parent, 'presolve', _cls.off_on_constants),
        qcpduals=_cls.NumParameter(env, _auto.PreprocessingQCPDuals, parent, 'qcpduals', _cls.qcpduals_constants),
        qpmakepsd=_cls.NumParameter(env, _auto.PreprocessingQPMakePSD, parent, 'qpmakepsd', _cls.off_on_constants),
        qtolin=_cls.NumParameter(env, _auto.PreprocessingQToLin, parent, 'qtolin', _cls.auto_off_on_constants),
        reduce=_cls.NumParameter(env, _auto.PreprocessingReduce, parent, 'reduce', _cls.prered_constants),
        reformulations=_cls.NumParameter(env, _auto.PreprocessingReformulations, parent, 'reformulations', _cls.prereform_constants),
        relax=_cls.NumParameter(env, _auto.PreprocessingRelax, parent, 'relax', _cls.auto_off_on_constants),
        repeatpresolve=_cls.NumParameter(env, _auto.PreprocessingRepeatPresolve, parent, 'repeatpresolve', _cls.repeatpre_constants),
        sos1reform=_cls.NumParameter(env, _auto.PreprocessingSOS1Reform, parent, 'sos1reform', _cls.sos1reform_constants),
        sos2reform=_cls.NumParameter(env, _auto.PreprocessingSOS2Reform, parent, 'sos2reform', _cls.sos2reform_constants),
        symmetry=_cls.NumParameter(env, _auto.PreprocessingSymmetry, parent, 'symmetry', _cls.sym_constants),
    )

def read_members(env, parent):
    """Problem read parameters."""
    return dict(
        _name="read",
        help=lambda: "Problem read parameters.",
        constraints=_cls.NumParameter(env, _auto.ReadConstraints, parent, 'constraints'),
        datacheck=_cls.NumParameter(env, _auto.ReadDataCheck, parent, 'datacheck', _cls.datacheck_constants),
        fileencoding=_cls.StrParameter(env, _auto.ReadFileEncoding, parent, 'fileencoding'),
        nonzeros=_cls.NumParameter(env, _auto.ReadNonzeros, parent, 'nonzeros'),
        qpnonzeros=_cls.NumParameter(env, _auto.ReadQPNonzeros, parent, 'qpnonzeros'),
        scale=_cls.NumParameter(env, _auto.ReadScale, parent, 'scale', _cls.scale_constants),
        variables=_cls.NumParameter(env, _auto.ReadVariables, parent, 'variables'),
        warninglimit=_cls.NumParameter(env, _auto.ReadWarningLimit, parent, 'warninglimit'),
    )

def sifting_members(env, parent):
    """Parameters for sifting optimization."""
    return dict(
        _name="sifting",
        help=lambda: "Parameters for sifting optimization.",
        algorithm=_cls.NumParameter(env, _auto.SiftingAlgorithm, parent, 'algorithm', _cls.sift_alg_constants),
        simplex=_cls.NumParameter(env, _auto.SiftingSimplex, parent, 'simplex', _cls.off_on_constants),
        display=_cls.NumParameter(env, _auto.SiftingDisplay, parent, 'display', _cls.display_constants),
        iterations=_cls.NumParameter(env, _auto.SiftingIterations, parent, 'iterations'),
    )

def simplex_limits_members(env, parent):
    """Limits for simplex optimization."""
    return dict(
        _name="limits",
        help=lambda: "Limits for simplex optimization.",
        iterations=_cls.NumParameter(env, _auto.SimplexLimitsIterations, parent, 'iterations'),
        lowerobj=_cls.NumParameter(env, _auto.SimplexLimitsLowerObj, parent, 'lowerobj'),
        perturbation=_cls.NumParameter(env, _auto.SimplexLimitsPerturbation, parent, 'perturbation'),
        singularity=_cls.NumParameter(env, _auto.SimplexLimitsSingularity, parent, 'singularity'),
        upperobj=_cls.NumParameter(env, _auto.SimplexLimitsUpperObj, parent, 'upperobj'),
    )

def simplex_perturbation_members(env, parent):
    """Perturbation controls."""
    return dict(
        _name="perturbation",
        help=lambda: "Perturbation controls.",
        constant=_cls.NumParameter(env, _auto.SimplexPerturbationConstant, parent, 'constant'),
        indicator=_cls.NumParameter(env, _auto.SimplexPerturbationIndicator, parent, 'indicator', _cls.off_on_constants),
    )

def simplex_tolerances_members(env, parent):
    """Numerical tolerances for simplex optimization."""
    return dict(
        _name="tolerances",
        help=lambda: "Numerical tolerances for simplex optimization.",
        feasibility=_cls.NumParameter(env, _auto.SimplexTolerancesFeasibility, parent, 'feasibility'),
        markowitz=_cls.NumParameter(env, _auto.SimplexTolerancesMarkowitz, parent, 'markowitz'),
        optimality=_cls.NumParameter(env, _auto.SimplexTolerancesOptimality, parent, 'optimality'),
    )

def simplex_members(env, parent):
    """Parameters for primal and dual simplex optimizations."""
    return dict(
        _name="simplex",
        help=lambda: "Parameters for primal and dual simplex optimizations.",
        crash=_cls.NumParameter(env, _auto.SimplexCrash, parent, 'crash'),
        dgradient=_cls.NumParameter(env, _auto.SimplexDGradient, parent, 'dgradient', _cls.dual_pricing_constants),
        display=_cls.NumParameter(env, _auto.SimplexDisplay, parent, 'display', _cls.display_constants),
        dynamicrows=_cls.NumParameter(env, _auto.SimplexDynamicRows, parent, 'dynamicrows'),
        limits=_cls.ParameterGroup(env, simplex_limits_members, parent),
        perturbation=_cls.ParameterGroup(env, simplex_perturbation_members, parent),
        pgradient=_cls.NumParameter(env, _auto.SimplexPGradient, parent, 'pgradient', _cls.primal_pricing_constants),
        pricing=_cls.NumParameter(env, _auto.SimplexPricing, parent, 'pricing'),
        refactor=_cls.NumParameter(env, _auto.SimplexRefactor, parent, 'refactor'),
        tolerances=_cls.ParameterGroup(env, simplex_tolerances_members, parent),
    )

def tune_members(env, parent):
    """Parameters for parameter tuning."""
    return dict(
        _name="tune",
        help=lambda: "Parameters for parameter tuning.",
        dettimelimit=_cls.NumParameter(env, _auto.TuneDetTimeLimit, parent, 'dettimelimit'),
        display=_cls.NumParameter(env, _auto.TuneDisplay, parent, 'display', _cls.tune_display_constants),
        measure=_cls.NumParameter(env, _auto.TuneMeasure, parent, 'measure', _cls.measure_constants),
        repeat=_cls.NumParameter(env, _auto.TuneRepeat, parent, 'repeat'),
        timelimit=_cls.NumParameter(env, _auto.TuneTimeLimit, parent, 'timelimit'),
    )

def root_members(env, parent):
    """CPLEX parameter hierarchy."""
    return dict(
        _name="parameters",
        help=lambda: "CPLEX parameter hierarchy.",
        advance=_cls.NumParameter(env, _auto.setAdvance, parent, 'advance', _cls.advance_constants),
        barrier=_cls.ParameterGroup(env, barrier_members, parent),
        benders=_cls.ParameterGroup(env, benders_members, parent),
        clocktype=_cls.NumParameter(env, _auto.setClockType, parent, 'clocktype', _cls.clocktype_constants),
        conflict=_cls.ParameterGroup(env, conflict_members, parent),
        cpumask=_cls.StrParameter(env, _auto.setCPUmask, parent, 'cpumask'),
        dettimelimit=_cls.NumParameter(env, _auto.setDetTimeLimit, parent, 'dettimelimit'),
        emphasis=_cls.ParameterGroup(env, emphasis_members, parent),
        feasopt=_cls.ParameterGroup(env, feasopt_members, parent),
        lpmethod=_cls.NumParameter(env, _auto.setLPMethod, parent, 'lpmethod', _cls.alg_constants),
        mip=_cls.ParameterGroup(env, mip_members, parent),
        multiobjective=_cls.ParameterGroup(env, multiobjective_members, parent),
        network=_cls.ParameterGroup(env, network_members, parent),
        optimalitytarget=_cls.NumParameter(env, _auto.setOptimalityTarget, parent, 'optimalitytarget', _cls.optimalitytarget_constants),
        output=_cls.ParameterGroup(env, output_members, parent),
        parallel=_cls.NumParameter(env, _auto.setParallel, parent, 'parallel', _cls.par_constants),
        paramdisplay=_cls.NumParameter(env, _auto.setParamDisplay, parent, 'paramdisplay', _cls.off_on_constants),
        preprocessing=_cls.ParameterGroup(env, preprocessing_members, parent),
        qpmethod=_cls.NumParameter(env, _auto.setQPMethod, parent, 'qpmethod', _cls.qp_alg_constants),
        randomseed=_cls.NumParameter(env, _auto.setRandomSeed, parent, 'randomseed'),
        read=_cls.ParameterGroup(env, read_members, parent),
        record=_cls.NumParameter(env, _auto.setRecord, parent, 'record', _cls.off_on_constants),
        sifting=_cls.ParameterGroup(env, sifting_members, parent),
        simplex=_cls.ParameterGroup(env, simplex_members, parent),
        solutiontype=_cls.NumParameter(env, _auto.setSolutionType, parent, 'solutiontype', _cls.solutiontype_constants),
        threads=_cls.NumParameter(env, _auto.setThreads, parent, 'threads'),
        timelimit=_cls.NumParameter(env, _auto.setTimeLimit, parent, 'timelimit'),
        tune=_cls.ParameterGroup(env, tune_members, parent),
        workdir=_cls.StrParameter(env, _auto.setWorkDir, parent, 'workdir'),
        workmem=_cls.NumParameter(env, _auto.setWorkMem, parent, 'workmem'),
    )
