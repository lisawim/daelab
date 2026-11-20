from abc import ABC, abstractmethod
import copy

from daelab.base.problems.dae_problem import DAEProblem


class Method(ABC):
    """
    Base class of time-stepping methods.

    A method is instantiated by the required parameters (e.g., ``num_nodes``,
    ``quad_type``, ``QI_type``), and is tied to a problem afterwards realized
    through ``setup``. The `TimeStepper` ties the method to its problem using
    ``bind`` allowing that one instance of the method can be used for several
    problems.

    Notes
    -----
    A subclass has to implement ``setup``: it stores the problem as attribute
    and creates everything that depends on it (e.g., the `NodeList`s for the
    solution ``u`` and the right-hand side ``f``).

    Attributes
    ----------
    problem : DAEProblem or None
        The problem instance. Default is ``None``. In ``setup``, the problem is
        stored as ``self.problem``. Calling ``_require_setup`` in `TimeStepper`,
        a ``RuntimeError`` is raised if ``self.problem`` is still ``None``. That
        should garantuee that the problem will be stored as attribute.
    """

    def __init__(self):
        """Initialization routine."""
        self.problem = None

    @abstractmethod
    def setup(self, problem: DAEProblem):
        """
        Ties the method to ``problem`` and creates the data that depends on it.
        It is called by ``bind``.

        Parameters
        ----------
        problem : DAEProblem
            The problem instance.

        Raises
        ------
        NotImplementedError
            If the subclasses does not implemented the method.
        """
        raise NotImplementedError("Subclasses must implement setup(problem).")

    def bind(self, problem: DAEProblem):
        """
        Creates a copy of the instantiated method, and sets up ``problem``.

        Binding creates a shallow copy, then calls ``setup(problem)`` to
        sets up the problem for this copy and returns it. Thus, one
        configuration of a method can be bound to several problems or
        timesteppers, and every bound copy has its own ``problem`` and its own
        data that is created in ``setup``. Be aware that only the copy
        created in ``bind`` has self.problem that is not None. The original
        instance still has ``self.problem`` stored as ``None``.

        Parameters
        ----------
        problem : DAEProblem
            Problem the returned method is set up for.

        Returns
        -------
        bound : Method
            New object of the same type as ``self`` with ``problem`` set and the
            problem-dependent data created.
        """
        bound = copy.copy(self)
        bound.setup(problem)
        return bound

    def _require_setup(self):
        """
        Checks that the method is set up for a problem. Be aware that only the
        copy created in ``bind`` has self.problem that is not None. The original
        instance still has ``self.problem`` stored as ``None``.

        Raises
        ------
        RuntimeError
            If problem is not an attribute of the method instance.
        """
        if self.problem is None:
            raise RuntimeError(
                f"{type(self).__name__} is not set up. Use TimeStepper(method, problem)"
                " or method.bind(problem)."
            )
