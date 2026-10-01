from ._solver import BaseSolver
from .det_solver import DetSolver

TASKS: dict[str, type[BaseSolver]] = {
    "detection": DetSolver,
}
