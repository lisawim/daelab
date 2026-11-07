from daelab.base.datatypes.solution import Solution


class RHS(Solution):
    """
    Right-hand side of a fullimplicit DAE problem at all collocation nodes.

    A fully-implicit DAE is given in residual form F(t, u, u') = 0. The class
    stores the residual F at each collocation node. It has the same structure
    as `Solution`, i.e., ``F[m]`` is the residual at node ``m`` and has shape
    ``nvars``, since a DAE has as many equations as unknowns. The separate class
    only marks the object as a right-hand side and distinguishes it from a
    solution.

    All parameters, attributes and methods are inherited from `Solution`.

    Parameters
    ----------
    num_nodes : int
        Number of collocation nodes.
    nvars : tuple of integers
        Number of unknowns (first entry), and spatial dimensions shared by all
        unknowns (remaining) entries.
    dtype : data-type, optional
        Data type of the entries. Default is float.

    Notes
    -----
    Solutions and right-hand sides can be combined in arithmetic expressions,
    e.g., for SDC. The result has the type of the left operand.

    Examples
    --------
    >>> F = RHS(num_nodes=3, nvars=(2, 4))
    >>> F[1].shape
    (2, 4)
    >>> F[1] = 1.0
    >>> u = Solution(num_nodes=3, nvars=(2, 4))
    >>> u[:] = 2.0
    >>> w = u + 0.5 * F
    >>> type(w).__name__, w[1][0, 0]
    'Solution', 2.5
    >>> type(2 * F).__name__
    'RHS'
    """
