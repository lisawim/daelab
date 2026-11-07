import numpy as np

from daelab.base.datatypes.semiexplicit_solution import PartitionedContainer
from daelab.base.datatypes.rhs import RHS


class IMEXRHS(PartitionedContainer):
    """
    Right-hand side for imex-splitting for each collocation node.

    The right-hand side of a problem can be splitted into implicit and explicit
    parts. Each component is a `RHS` and is available as attribute ``impl`` and
    ``expl``. The class provides access to all parts at a node, assignment,
    copying, and arithmetic, where every operation is applied part by part.

    Parameters
    ----------
    num_nodes : int
        Number of collocation nodes.
    nvars : tuple of integers
        Number of unknowns (first entry), and spatial dimensions shared by all
        unknowns (remaining) entries.
    dtype : data-type, optional
        Data type of the entries. Default is float.

    Attributes
    ----------
    num_nodes : int
        Number of collocation nodes.
    nvars : tuple of integers
        Number of unknowns (first entry), and spatial dimensions (remaining
        entries) shared by all unknowns.
    dtype : data-type, optional
        Data type of the entries. Default is float.
    impl : RHS
       Right-hand side for implicit treatment.
    expl : RHS
        Right-hand side for implicit treatment.

    Notes
    -----
    - For implicit and explicit parts, the right-hand side at each collocation
      node is stored in a list that has ``num_nodes``entries, each of them is
      of shape ``nvars`` and of data type ``dtype``. Using ``f[m]`` the
      right-hand side ``f`` at node ``m`` is accessed at both parts, ``impl``
      and ``expl``. The implicit and explicit parts of the right-hand side at
      node ``m`` can be accessed using ``f.impl[m]`` and ``f.expl[m]``,
      respectively.

    Examples
    --------
    >>> f = IMEXRHS(num_nodes=4, nvars=(2, 4))
    >>> len(f.impl), len(f.expl)
    4, 4
    >>> f.impl[1].shape
    (2, 4)
    >>> f.impl[1] = 2 * np.ones((2, 4))
    >>> f.expl[1] = np.ones((2, 4))
    >>> g1 = f.impl[1] + 10 * f.expl[1]
    [[12. 12. 12. 12.]
    [12. 12. 12. 12.]]
    >>> f.full_rhs
    RHS(num_nodes=4, nvars=(2, 4), dtype=float64)
    """

    _components = ("impl", "expl")

    def __init__(self, num_nodes, nvars, dtype=float):
        """Initialization routine."""
        self.num_nodes = num_nodes
        self.nvars = nvars
        self.dtype = dtype
        self.impl = RHS(num_nodes=self.num_nodes, nvars=self.nvars, dtype=self.dtype)
        self.expl = RHS(num_nodes=self.num_nodes, nvars=self.nvars, dtype=self.dtype)

    @property
    def full_rhs(self):
        """
        Returns the full right-hand side.

        Returns
        -------
        RHS
            Full right-hand side as sum of implicit and explicit parts.
        """
        return self.impl + self.expl

    def __repr__(self):
        """
        Returns the class with inputs as string.

        Returns
        str
            Class description.
        """
        return (
            f"{type(self).__name__}(num_nodes={self.num_nodes}, "
            f"nvars={self.nvars}, dtype={np.dtype(self.dtype)})"
        )
