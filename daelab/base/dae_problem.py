class DAEProblem:
    def __init__(
        self,
        t_span: tuple[float, float] | None = None,
        name: str | None = None,
    ):

        if t_span is not None:
            if len(t_span) != 2:
                raise ValueError("t_span is a tuple containing two values!")

            t0, tend = t_span

            if t0 >= tend:
                raise ValueError("tend must be greater than t0!")

        self._t_span = None if t_span is None else t_span

        self.name = None if name is None else name

    @property
    def t_span(self) -> tuple[float, float] | None:
        return self._t_span

    @property
    def t0(self) -> float | None:
        return None if self._t_span is None else self._t_span[0]

    @property
    def t_end(self) -> float | None:
        return None if self._t_span is None else self._t_span[1]
