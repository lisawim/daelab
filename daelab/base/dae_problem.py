class DAEProblem:
    """Base class for differential-algebraic equations problems."""
    def __init__(
        self,
        diff_index: int | None = None,
        t_span: tuple[float, float] | None = None,
        name: str | None = None,
    ):
        """
        Initializes a DAE Problem.

        Parameters
        ----------
        diff_index : int or None
            Differentiation index of the DAE problem.
        t_span : tuple[float, float] or None
            Time interval (t0, t_end) on which the problem is defined.
            If None, no specific time interval is prescribed.
        name : str or None
            Name of the problem. If None, the class name is used.

        Raises
        ------
        TypeError
            If diff_index is bool or not an integer.
        ValueError
            If t_span does not contain exactly two values or
            if t_end is not greater than t0.
        """

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
    def validate_nvars(nvars: tuple[tuple[int, ...], ...]):
        if not isinstance(nvars, tuple):
            raise TypeError(f"nvars must be a tuple.")

        if not nvars:
            raise ValueError(f"nvars must not be empty.")

        for dimensions in nvars:
            if not isinstance(dimensions, tuple):
                raise TypeError(
                    f"Each entry of nvars must be a tuple."
                )

            if not dimensions:
                raise ValueError(
                    f"Each entry of nvars must not be empty."
                )

            if any(
                not isinstance(ndim, int) or ndim < 1 for ndim in dimensions
            ):
                raise ValueError(
                    f"nvars must contain only positive integers."
                )

    @property
    def t_span(self) -> tuple[float, float] | None:
        return self._t_span

    @property
    def t0(self) -> float | None:
        return None if self._t_span is None else self._t_span[0]

    @property
    def t_end(self) -> float | None:
        return None if self._t_span is None else self._t_span[1]
