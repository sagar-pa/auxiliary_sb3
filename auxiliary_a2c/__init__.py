from auxiliary_a2c.a2c import AuxiliaryA2C
from auxiliary_a2c.policies import AuxiliaryMlpPolicy, AuxiliaryCnnPolicy
from auxiliary_a2c.objectives import AuxiliaryObjective, DecomposedReturnsObjective

__all__ = ["AuxiliaryMlpPolicy", "AuxiliaryCnnPolicy", "AuxiliaryA2C",
    "AuxiliaryObjective", "DecomposedReturnsObjective"]