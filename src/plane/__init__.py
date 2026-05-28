"""mini-PlanE: a minimal implementation of PlanE (Representation Learning over Planar Graphs)."""

from plane.data import DataPlanE, planar_preprocess
from plane.model.model import PlanE

__version__ = "0.1.0"
__all__ = ["PlanE", "planar_preprocess", "DataPlanE"]
