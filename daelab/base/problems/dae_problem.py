class DAEProblem:
    """
    Base class for differential-algebraic equation problems.

    Parameters
    ----------
    dtype : type
        Datatype of the solution. Default is `float`.
    diff_index : int or None
        Differentiation index of the DAE problem.
    t_span : tuple[float, float] or None
        Time interval (t0, t_end) on which the problem is defined.
        If None, no specific time interval is prescribed.
    name : str or None
        Name of the problem. If None, the class name is used.

    Attributes
    ----------
    dtype : type
        Datatype of the solution.
    diff_index : int or None
        Differentiation index.
    name : str or None
        Name of the problem.

    Raises
    ------
    TypeError
        If `diff_index` is bool or not an integer.
    ValueError
        If `t_span` does not contain exactly two values or if `t_end` is not
        greater than `t0`.
    """

    def __init__(
        self,
        dtype: type = float,
        diff_index: int | None = None,
        t_span: tuple[float, float] | None = None,
        name: str | None = None,
    ):
        """Initializes a DAE Problem."""

        self.dtype = dtype

        if diff_index is not None:
            if isinstance(diff_index, bool) or not isinstance(diff_index, int):
                raise TypeError("diff_index must be an integer or None!")

            if diff_index < 1:
                raise ValueError("diff_index must be equal to or greater than one!")

        self.diff_index = diff_index

        if t_span is not None:
            if len(t_span) != 2:
                raise ValueError("t_span is a tuple containing two values!")

            t0, t_end = t_span

            if t0 >= t_end:
                raise ValueError("t_end must be greater than t0!")

        self._t_span = None if t_span is None else t_span

        self.name = self.__class__.__name__ if name is None else name

    @staticmethod
    def validate_nvars(nvars: tuple[int, ...]):
        """
        Validates that `nvars` is of correct shape and entries.

        Parameters
        ----------
        nvars : tuple of integers
            Number of unknowns and spatial dimensions to check.

        Raises
        ------
        TypeError
            If `nvars` is not a tuple.
        ValueError
            If `nvars` is None or `nvars` contains nonpositive integers.
        """
        if not isinstance(nvars, tuple):
            raise TypeError("nvars must be a tuple.")

        if not nvars:
            raise ValueError("nvars must not be empty.")

        if any(not isinstance(n, int) or n < 1 for n in nvars):
            raise ValueError("nvars must contain only positive integers.")

    @property
    def t_span(self) -> tuple[float, float] | None:
        """
        Property `t_span`.

        Returns
        -------
        tuple
            The time interval.
        """
        return self._t_span

    @property
    def t0(self) -> float | None:
        """
        Property `t0`.

        Returns
        -------
        float or None
            The initial time.
        """
        return None if self._t_span is None else self._t_span[0]

    @property
    def t_end(self) -> float | None:
        """
        Property `t_end`.

        Returns
        -------
        float or None
            The end time.
        """
        return None if self._t_span is None else self._t_span[1]
