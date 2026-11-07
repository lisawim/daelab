from daelab.base.datatypes.semiexplicit_solution import SemiexplicitSolution
from daelab.base.datatypes.rhs import RHS
from daelab.base.datatypes.imex_rhs import IMEXRHS


class SemiexplicitRHS(SemiexplicitSolution):
    """
    Right-hand side with splitting into differential and algebraic equation at
    each collocation node.

    Semi-explicit DAEs have differential equations subject to algebraic
    constraints. Each equation type component is a `RHS` and is available as
    attribute ``diff`` and ``alg``. In this class, the right-hand side of the
    differential equation  and the algebraic constraints  at each node are
    stored as a list of numpy arrays, and each array is of shape ``nvars_diff``
    and ``nvars_alg``, respectively. The class provides access to all parts at
    a node, assignment, copying, and arithmetic, where every operation is
    applied part by part. In addition, the right-hand side of the differential
    equation ``f.diff`` can use an imex-splitting by setting ``imex_diff=True``.

    Parameters
    ----------
    num_nodes : int
        Number of collocation nodes.
    nvars_diff : tuple of integers
        Number of unknowns of differential variables (first entry), and spatial
        dimensions shared by all unknowns (remaining) entries.
    nvars_diff : tuple of integers
        Number of unknowns of algebraic variables (first entry), and spatial
        dimensions shared by all unknowns (remaining) entries.
    dtype : data-type, optional
        Data type of the entries. Default is float.
    imex_diff : bool, optional
        Indicates whether right-hand side of differential equation should use an
        imex splitting. Default is False.

    Attributes
    ----------
    num_nodes : int
        Number of collocation nodes.
    nvars_diff : tuple of integers
        Shape of solution of differential variables at single collocation node.
    nvars_alg : tuple of integers
        Shape of solution of algebraic variables at single collocation node.
    dtype : data-type, optional
        Data type of the entries. Default is float.
    imex_diff : bool, optional
        Flat to choose imex splitting.
    diff : RHS or IMEXRHS
        Right-hand side of differential equation.
    alg : RHS
        Right-hand side of algebraic equation.

    Examples
    --------
    >>> f = SemiexplicitRHS(
    ...     num_nodes=3, nvars_diff=(2, 4), nvars_alg=(1, 4), imex_diff=True
    ... )
    >>> f.diff
    IMEXRHS(num_nodes=3, nvars=(2, 4), dtype=float64)
    >>> f.diff.impl
    RHS(num_nodes=3, nvars=(2, 4), dtype=float64)
    >>> f.diff.impl[2] = 0.25 * np.ones((2, 4))
    >>> f.diff.expl[2] = 0.33 * np.ones((2, 4))
    >>> f.diff.full_rhs[2]
    [[0.58 0.58 0.58 0.58]
    [0.58 0.58 0.58 0.58]]
    """

    def __init__(self, num_nodes, nvars_diff, nvars_alg, dtype=float, imex_diff=False):
        super().__init__(num_nodes, nvars_diff, nvars_alg, dtype)
        self.imex_diff = imex_diff
        self.diff = (
            IMEXRHS(self.num_nodes, self.nvars_diff, self.dtype)
            if self.imex_diff
            else RHS(self.num_nodes, self.nvars_diff, self.dtype)
        )
        self.alg = RHS(self.num_nodes, self.nvars_alg, self.dtype)
