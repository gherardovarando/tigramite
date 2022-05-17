from typing import Optional, Literal 

from tigramite.independence_tests import ParCorr, CMIknn
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
import numpy as np

TESTS = Literal['ParrCorr', 'CMIknn']
FDR = Literal['none', 'fdr_bh']
def pcmci(data: np.ndarray,
          cond_ind_test: TESTS ='ParrCorr', 
          tau_min: int = 0,
          tau_max: int = 1,
          pc_alpha: float = 0.05,
          max_conds_dim: Optional[int] = None,
          max_combinations: int = 1,
          max_conds_py: Optional[int] = None,
          max_conds_px: Optional[int] = None,
          alpha_level: float = 0.05,
          fdr_method: FDR = 'fdr_bh') -> np.ndarray:
    r"""PCMCI causal discovery for time series datasets.

    PCMCI is a causal discovery framework for large-scale time series
    datasets. This class contains several methods. The standard PCMCI method
    addresses time-lagged causal discovery and is described in [1]_ where
    also further sub-variants are discussed. Lagged as well as contemporaneous
    causal discovery is addressed with PCMCIplus and described in [5]_. See the
    tutorials for guidance in applying these methods.

    PCMCI has:

    * different conditional independence tests adapted to linear or
      nonlinear dependencies, and continuously-valued or discrete data (
      implemented in ``tigramite.independence_tests``)
    * (mostly) hyperparameter optimization
    * easy parallelization (separate script)
    * handling of masked time series data
    * false discovery control and confidence interval estimation


    Notes
    -----

    In the PCMCI framework, the dependency structure of a set of time series
    variables is represented in a *time series graph* as shown in the Figure.
    The nodes of a time series graph are defined as the variables at
    different times and a link indicates a conditional dependency that can be
    interpreted as a causal dependency under certain assumptions (see paper).
    Assuming stationarity, the links are repeated in time. The parents
    :math:`\mathcal{P}` of a variable are defined as the set of all nodes
    with a link towards it (blue and red boxes in Figure).

    The different PCMCI methods estimate causal links by iterative
    conditional independence testing. PCMCI can be flexibly combined with
    any kind of conditional independence test statistic adapted to the kind
    of data (continuous or discrete) and its assumed dependency types.
    These are available in ``tigramite.independence_tests``.

    NOTE: MCI test statistic values define a particular measure of causal
    strength depending on the test statistic used. For example, ParCorr()
    results in normalized values between -1 and 1. However, if you are 
    interested in quantifying causal effects, i.e., the effect of
    hypothetical interventions, you may better look at the causal effect 
    estimation functionality of Tigramite.


    The PCMCI causal discovery method is comprehensively described in [
    1]_, where also analytical and numerical results are presented. Here
    we briefly summarize the method.

    PCMCI estimates time-lagged causal links by a two-step procedure:

    1.  Condition-selection: For each variable :math:`j`, estimate a
        *superset* of parents :math:`\\tilde{\mathcal{P}}(X^j_t)` with the
        iterative PC1 algorithm, implemented as ``run_pc_stable``. The
        condition-selection step reduces the dimensionality and avoids
        conditioning on irrelevant variables.

    2.  *Momentary conditional independence* (MCI)

    .. math:: X^i_{t-\\tau} \perp X^j_{t} | \\tilde{\\mathcal{P}}(
              X^j_t), \\tilde{\mathcal{P}}(X^i_{t-\\tau})

    here implemented as ``run_mci``. This step estimates the p-values and
    test statistic values for all links accounting for common drivers,
    indirect links, and autocorrelation.

    NOTE: MCI test statistic values define a particular measure of causal
    strength depending on the test statistic used. For example, ParCorr()
    results in normalized values between -1 and 1. However, if you are 
    interested in quantifying causal effects, i.e., the effect of
    hypothetical interventions, you may better look at the causal effect 
    estimation functionality of Tigramite.

    PCMCI can be flexibly combined with any kind of conditional
    independence test statistic adapted to the kind of data (continuous
    or discrete) and its assumed dependency types. These are available in
    ``tigramite.independence_tests``.

    The main free parameters of PCMCI (in addition to free parameters of
    the conditional independence test statistic) are the maximum time
    delay :math:`\\tau_{\\max}` (``tau_max``) and the significance
    threshold in the condition-selection step :math:`\\alpha` (
    ``pc_alpha``). The maximum time delay depends on the application and
    should be chosen according to the maximum causal time lag expected in
    the complex system. We recommend a rather large choice that includes
    peaks in the ``get_lagged_dependencies`` function. :math:`\\alpha`
    should not be seen as a significance test level in the
    condition-selection step since the iterative hypothesis tests do not
    allow for a precise assessment. :math:`\\alpha` rather takes the role
    of a regularization parameter in model-selection techniques. If a
    list of values is given or ``pc_alpha=None``, :math:`\\alpha` is
    optimized using model selection criteria implemented in the respective
    ``tigramite.independence_tests``.

    Further optional parameters are discussed in [1]_.


    References
    ----------

    .. [1] J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, D. Sejdinovic,
           Detecting and quantifying causal associations in large nonlinear time 
           series datasets. Sci. Adv. 5, eaau4996 (2019) 
           https://advances.sciencemag.org/content/5/11/eaau4996

    .. [5] J. Runge,
           Discovering contemporaneous and lagged causal relations in 
           autocorrelated nonlinear time series datasets
           http://www.auai.org/~w-auai/uai2020/proceedings/579_main_paper.pdf

    Parameters
    ----------
    data : numpy array 
        numpy array of shape (observations T, variables N).
    cond_ind_test : string
        conditional independence test to be used 
    tau_min : int, optional (default: 0)
        Minimum time lag to test. Note that zero-lags are undirected.
    tau_max : int, optional (default: 1)
        Maximum time lag. Must be larger or equal to tau_min.
    pc_alpha : float, optional (default: 0.05)
        Significance level in algorithm.
    max_conds_dim : int, optional (default: None)
        Maximum number of conditions to test. If None is passed, this number
        is unrestricted.
    max_combinations : int, optional (default: 1)
        Maximum number of combinations of conditions of current cardinality
        to test. Defaults to 1 for PC_1 algorithm. For original PC algorithm
        a larger number, such as 10, can be used.
    max_conds_py : int, optional (default: None)
        Maximum number of conditions of Y to use. If None is passed, this
        number is unrestricted.
    max_conds_px : int, optional (default: None)
        Maximum number of conditions of Z to use. If None is passed, this
        number is unrestricted.
    alpha_level : float, optional (default: 0.05)
        Significance level at which the p_matrix is thresholded to 
        get graph.
    fdr_method : str, optional (default: 'fdr_bh')
        Correction method, currently implemented is Benjamini-Hochberg
        False Discovery Rate method. 
    """

    dataframe = pp.DataFrame(data)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
    results = pcmci.run_pcmci(tau_min = tau_min, tau_max=tau_max, pc_alpha=pc_alpha,
                              max_conds_dim = max_conds_dim, max_combinations = max_combinations,
                              max_conds_py = max_conds_py, max_conds_px = max_conds_px,
                              alpha_level = alpha_level, fdr_method = fdr_method)
    val_matrix = np.sum(results['val_matrix'], axis=2)
    return val_matrix 


