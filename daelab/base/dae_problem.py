class DAEProblem:
    """Base class for differential-algebraic equations problems."""
    def __init__(
        self,
        t_span: tuple[float, float] | None = None,
        name: str | None = None,
    ):
        """
        Initializes a DAE Problem.

        Parameters
        ----------
        t_span : tuple[float, float] or None
            Time interval (t0, t_end) on which the problem is defined.
            If None, no specific time interval is prescribed.
        name : str or None
            Name of the problem. If None, the class name is used.

        Raises
        ------
        ValueError
            If t_span does not contain exactly two values or
            if t_end is not greater than t0.
        """

        if t_span is not None:
            if len(t_span) != 2:
                raise ValueError("t_span is a tuple containing two values!")

            t0, t_end = t_span

            if t0 >= t_end:
                raise ValueError("t_end must be greater than t0!")

        self._t_span = None if t_span is None else t_span

        self.name = self.__class__.__name__ if name is None else name

    @property
    def t_span(self) -> tuple[float, float] | None:
        return self._t_span

    @property
    def t0(self) -> float | None:
        return None if self._t_span is None else self._t_span[0]

    @property
    def t_end(self) -> float | None:
        return None if self._t_span is None else self._t_span[1]
