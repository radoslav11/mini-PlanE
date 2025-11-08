"""
mini-PlanE: A simplified interface for PlanE (Representation Learning over Planar Graphs)

This package provides an easy-to-use implementation of PlanE without complex configuration flags.
Perfect for users new to planar graph neural networks.
"""

from   plane.data               import DataPlanE, planar_preprocess
from   plane.model.layers       import PlaneLayer
from   plane.model.model        import PlanE, SimplePlanE

__version__ = "0.1.0"
__all__ = ["PlanE", "SimplePlanE", "PlaneLayer", "planar_preprocess", "DataPlanE"]
