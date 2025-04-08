# None attacks

# Linf attacks
from attacks.fgsm import FGSM
from attacks.mifgsm import MIFGSM
from attacks.pgd import PGD


__version__ = "3.5.1"
__all__ = [

    "FGSM",
    "MIFGSM",
    "PGD"

]
__wrapper__ = [
    "LGV",
    "MultiAttack",
]
