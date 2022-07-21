"""Molecules for testing properties."""

from __future__ import annotations
import torch

from xtbml.typing import Tensor, Molecule
from xtbml.utils import symbol2number


class Record(Molecule):
    """Format of reference records (calculated with xTB 6.5.1)."""

    overlap: Tensor
    """Overlap matrix for GFN1-xTB."""

    density: Tensor
    """Density matrix for GFN1-xTB."""

    wiberg: Tensor
    """Reference values for Wiberg bond orders."""


samples: dict[str, Record] = {
    "H2": {
        "numbers": symbol2number(["H", "H"]),
        "positions": torch.tensor(
            [
                0.00000000000000,
                0.00000000000000,
                -0.70252931147690,
                0.00000000000000,
                0.00000000000000,
                0.70252931147690,
            ],
        ).reshape((-1, 3)),
        "density": torch.tensor(
            [
                0.59854788165593265,
                3.1933160084198133e-003,
                0.59854788165593242,
                3.1933160084199668e-003,
                3.1933160084198133e-003,
                1.7036677335518522e-005,
                3.1933160084198125e-003,
                1.7036677335519342e-005,
                0.59854788165593242,
                3.1933160084198125e-003,
                0.59854788165593231,
                3.1933160084199664e-003,
                3.1933160084199668e-003,
                1.7036677335519342e-005,
                3.1933160084199664e-003,
                1.7036677335520162e-005,
            ]
        ).reshape((4, 4)),
        "overlap": torch.tensor(
            [
                1.0000000000000000,
                0.0000000000000000,
                0.66998297071517465,
                6.5205745680663674e-002,
                0.0000000000000000,
                1.0000000000000000,
                6.5205745680664340e-002,
                0.10264305272175234,
                0.66998297071517465,
                6.5205745680664340e-002,
                1.0000000000000000,
                0.0000000000000000,
                6.5205745680663674e-002,
                0.10264305272175234,
                0.0000000000000000,
                1.0000000000000000,
            ]
        ).reshape((4, 4)),
        "wiberg": torch.tensor(
            [
                [0.0000000000000000, 1.0000000000000007],
                [1.0000000000000007, 0.0000000000000000],
            ]
        ),
    },
    "LiH": {
        "numbers": symbol2number(["Li", "H"]),
        "positions": torch.tensor(
            [
                0.00000000000000,
                0.00000000000000,
                -1.50796743897235,
                0.00000000000000,
                0.00000000000000,
                1.50796743897235,
            ],
        ).reshape((-1, 3)),
        "density": torch.tensor(
            [
                0.20683869182353645,
                0.0000000000000000,
                0.0000000000000000,
                0.18337511649124147,
                0.43653303718019015,
                7.8111260740060754e-003,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.18337511649124147,
                0.0000000000000000,
                0.0000000000000000,
                0.16257322579116207,
                0.38701316392721402,
                6.9250397066456734e-003,
                0.43653303718019015,
                0.0000000000000000,
                0.0000000000000000,
                0.38701316392721402,
                0.92130292872059771,
                1.6485380751645420e-002,
                7.8111260740060754e-003,
                0.0000000000000000,
                0.0000000000000000,
                6.9250397066456734e-003,
                1.6485380751645420e-002,
                2.9498199783660944e-004,
            ]
        ).reshape((6, 6)),
        "overlap": torch.tensor(
            [
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.40560703697368278,
                -0.20099517781353604,
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000,
                0.46387245581706105,
                -7.5166937761294322e-002,
                0.40560703697368278,
                0.0000000000000000,
                0.0000000000000000,
                0.46387245581706105,
                1.0000000000000000,
                0.0000000000000000,
                -0.20099517781353604,
                0.0000000000000000,
                0.0000000000000000,
                -7.5166937761294322e-002,
                0.0000000000000000,
                1.0000000000000000,
            ]
        ).reshape((6, 6)),
        "wiberg": torch.tensor(
            [
                [0.0000000000000000, 0.92377265725501168],
                [0.92377265725501168, 0.0000000000000000],
            ]
        ),
    },
    "SiH4": {
        "numbers": symbol2number(["Si", "H", "H", "H", "H"]),
        "positions": torch.tensor(
            [
                0.00000000000000,
                -0.00000000000000,
                0.00000000000000,
                1.61768389755830,
                1.61768389755830,
                -1.61768389755830,
                -1.61768389755830,
                -1.61768389755830,
                -1.61768389755830,
                1.61768389755830,
                -1.61768389755830,
                1.61768389755830,
                -1.61768389755830,
                1.61768389755830,
                1.61768389755830,
            ],
        ).reshape((-1, 3)),
        "density": torch.tensor(
            [
                0.95266977814088072,
                -6.3188240095752813e-016,
                5.8478535363473266e-016,
                4.8228286168368446e-016,
                -1.5321944884146509e-031,
                2.0860667943490419e-025,
                1.6927390197181194e-017,
                -3.6499432127887788e-016,
                1.4516662171400914e-016,
                0.23137623597272830,
                6.7969218173179208e-003,
                0.23137623597272880,
                6.7969218173179781e-003,
                0.23137623597272938,
                6.7969218173181481e-003,
                0.23137623597272874,
                6.7969218173176095e-003,
                -6.3188240095752813e-016,
                0.34736135543141627,
                7.6327832942979512e-017,
                6.3143934525555778e-016,
                -1.0002303429944862e-017,
                1.0589079977801452e-018,
                -1.4224732503009818e-016,
                3.4694469519536142e-018,
                -0.11969117413304249,
                0.27325141025959709,
                -3.2189847434777753e-003,
                -0.27325141025959809,
                3.2189847434778108e-003,
                0.27325141025959804,
                -3.2189847434774951e-003,
                -0.27325141025959732,
                3.2189847434777475e-003,
                5.8478535363473266e-016,
                7.6327832942979512e-017,
                0.34736135543141561,
                4.1633363423443370e-016,
                2.2403404607595847e-016,
                5.6833727007981764e-019,
                1.2490009027033011e-016,
                -0.11969117413304239,
                3.2265856653168612e-016,
                0.27325141025959732,
                -3.2189847434778221e-003,
                -0.27325141025959776,
                3.2189847434775220e-003,
                -0.27325141025959648,
                3.2189847434780797e-003,
                0.27325141025959765,
                -3.2189847434777102e-003,
                4.8228286168368446e-016,
                6.3143934525555778e-016,
                4.1633363423443370e-016,
                0.34736135543141611,
                -2.3373850371649520e-016,
                -7.9186192927219019e-019,
                -0.11969117413304264,
                0.0000000000000000,
                -1.5959455978986625e-016,
                -0.27325141025959737,
                3.2189847434776113e-003,
                -0.27325141025959732,
                3.2189847434776677e-003,
                0.27325141025959720,
                -3.2189847434778061e-003,
                0.27325141025959748,
                -3.2189847434777714e-003,
                -1.5321944884146509e-031,
                -1.0002303429944862e-017,
                2.2403404607595847e-016,
                -2.3373850371649520e-016,
                3.0206292770933252e-031,
                8.6890495109000881e-034,
                8.0539862919355916e-017,
                -7.7195973591547266e-017,
                3.4465187990708944e-018,
                3.5223794868856983e-016,
                -4.1494701960557157e-018,
                1.5502300903535370e-017,
                -1.8262182087148494e-019,
                -3.6797454963937003e-016,
                4.3348521427647150e-018,
                2.3430004726511437e-019,
                -2.7601258372843535e-021,
                2.0860667943490419e-025,
                1.0589079977801452e-018,
                5.6833727007981764e-019,
                -7.9186192927219019e-019,
                8.6890495109000881e-034,
                5.9630660779252999e-036,
                2.7285385257704187e-019,
                -1.9583339970255441e-019,
                -3.6487058670001874e-019,
                1.9029879416593321e-018,
                -2.2417774809023393e-020,
                -6.5715329068212422e-019,
                7.7414677735717830e-021,
                -2.3701032611168600e-019,
                2.7920558318493872e-021,
                -1.0088241222071404e-018,
                1.1884257156908338e-020,
                1.6927390197181194e-017,
                -1.4224732503009818e-016,
                1.2490009027033011e-016,
                -0.11969117413304264,
                8.0539862919355916e-017,
                2.7285385257704187e-019,
                4.1242288301051075e-002,
                -9.0205620750793969e-017,
                2.9490299091605721e-017,
                9.4154924306018753e-002,
                -1.1091736528511399e-003,
                9.4154924306018170e-002,
                -1.1091736528511533e-003,
                -9.4154924306018475e-002,
                1.1091736528512088e-003,
                -9.4154924306018267e-002,
                1.1091736528511930e-003,
                -3.6499432127887788e-016,
                3.4694469519536142e-018,
                -0.11969117413304239,
                0.0000000000000000,
                -7.7195973591547266e-017,
                -1.9583339970255441e-019,
                -9.0205620750793969e-017,
                4.1242288301050957e-002,
                -1.2143064331837650e-016,
                -9.4154924306018475e-002,
                1.1091736528512101e-003,
                9.4154924306018323e-002,
                -1.1091736528511063e-003,
                9.4154924306018170e-002,
                -1.1091736528513016e-003,
                -9.4154924306018392e-002,
                1.1091736528511694e-003,
                1.4516662171400914e-016,
                -0.11969117413304249,
                3.2265856653168612e-016,
                -1.5959455978986625e-016,
                3.4465187990708944e-018,
                -3.6487058670001874e-019,
                2.9490299091605721e-017,
                -1.2143064331837650e-016,
                4.1242288301050950e-002,
                -9.4154924306017962e-002,
                1.1091736528511897e-003,
                9.4154924306018184e-002,
                -1.1091736528512021e-003,
                -9.4154924306018725e-002,
                1.1091736528510985e-003,
                9.4154924306018531e-002,
                -1.1091736528511876e-003,
                0.23137623597272830,
                0.27325141025959709,
                0.27325141025959732,
                -0.27325141025959737,
                3.5223794868856983e-016,
                1.9029879416593321e-018,
                9.4154924306018753e-002,
                -9.4154924306018475e-002,
                -9.4154924306017962e-002,
                0.70105339039614190,
                -5.9458540562029724e-003,
                -0.15875823851836182,
                4.1829885108348105e-003,
                -0.15875823851836068,
                4.1829885108354376e-003,
                -0.15875823851836154,
                4.1829885108348435e-003,
                6.7969218173179208e-003,
                -3.2189847434777753e-003,
                -3.2189847434778221e-003,
                3.2189847434776113e-003,
                -4.1494701960557157e-018,
                -2.2417774809023393e-020,
                -1.1091736528511399e-003,
                1.1091736528512101e-003,
                1.1091736528511897e-003,
                -5.9458540562029724e-003,
                1.3798398225090867e-004,
                4.1829885108350673e-003,
                1.8663135337776167e-005,
                4.1829885108347975e-003,
                1.8663135337773551e-005,
                4.1829885108347125e-003,
                1.8663135337776211e-005,
                0.23137623597272880,
                -0.27325141025959809,
                -0.27325141025959776,
                -0.27325141025959732,
                1.5502300903535370e-017,
                -6.5715329068212422e-019,
                9.4154924306018170e-002,
                9.4154924306018323e-002,
                9.4154924306018184e-002,
                -0.15875823851836182,
                4.1829885108350673e-003,
                0.70105339039614156,
                -5.9458540562027755e-003,
                -0.15875823851836079,
                4.1829885108344948e-003,
                -0.15875823851836132,
                4.1829885108347802e-003,
                6.7969218173179781e-003,
                3.2189847434778108e-003,
                3.2189847434775220e-003,
                3.2189847434776677e-003,
                -1.8262182087148494e-019,
                7.7414677735717830e-021,
                -1.1091736528511533e-003,
                -1.1091736528511063e-003,
                -1.1091736528512021e-003,
                4.1829885108348105e-003,
                1.8663135337776167e-005,
                -5.9458540562027755e-003,
                1.3798398225090531e-004,
                4.1829885108350499e-003,
                1.8663135337782306e-005,
                4.1829885108345807e-003,
                1.8663135337779104e-005,
                0.23137623597272938,
                0.27325141025959804,
                -0.27325141025959648,
                0.27325141025959720,
                -3.6797454963937003e-016,
                -2.3701032611168600e-019,
                -9.4154924306018475e-002,
                9.4154924306018170e-002,
                -9.4154924306018725e-002,
                -0.15875823851836068,
                4.1829885108347975e-003,
                -0.15875823851836079,
                4.1829885108350499e-003,
                0.70105339039614090,
                -5.9458540562030279e-003,
                -0.15875823851836146,
                4.1829885108347568e-003,
                6.7969218173181481e-003,
                -3.2189847434774951e-003,
                3.2189847434780797e-003,
                -3.2189847434778061e-003,
                4.3348521427647150e-018,
                2.7920558318493872e-021,
                1.1091736528512088e-003,
                -1.1091736528513016e-003,
                1.1091736528510985e-003,
                4.1829885108354376e-003,
                1.8663135337773551e-005,
                4.1829885108344948e-003,
                1.8663135337782306e-005,
                -5.9458540562030279e-003,
                1.3798398225091488e-004,
                4.1829885108349285e-003,
                1.8663135337777064e-005,
                0.23137623597272874,
                -0.27325141025959732,
                0.27325141025959765,
                0.27325141025959748,
                2.3430004726511437e-019,
                -1.0088241222071404e-018,
                -9.4154924306018267e-002,
                -9.4154924306018392e-002,
                9.4154924306018531e-002,
                -0.15875823851836154,
                4.1829885108347125e-003,
                -0.15875823851836132,
                4.1829885108345807e-003,
                -0.15875823851836146,
                4.1829885108349285e-003,
                0.70105339039614123,
                -5.9458540562030635e-003,
                6.7969218173176095e-003,
                3.2189847434777479e-003,
                -3.2189847434777102e-003,
                -3.2189847434777714e-003,
                -2.7601258372843505e-021,
                1.1884257156908338e-020,
                1.1091736528511930e-003,
                1.1091736528511694e-003,
                -1.1091736528511876e-003,
                4.1829885108348435e-003,
                1.8663135337776214e-005,
                4.1829885108347802e-003,
                1.8663135337779104e-005,
                4.1829885108347568e-003,
                1.8663135337777064e-005,
                -5.9458540562030635e-003,
                1.3798398225090480e-004,
            ]
        ).reshape(17, 17),
        "overlap": torch.tensor(
            [
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.42641210131950702,
                -0.14556746567207857,
                0.42641210131950702,
                -0.14556746567207857,
                0.42641210131950702,
                -0.14556746567207857,
                0.42641210131950702,
                -0.14556746567207857,
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.28196829739436963,
                6.4855251203039338e-002,
                -0.28196829739436963,
                -6.4855251203039338e-002,
                0.28196829739436963,
                6.4855251203039338e-002,
                -0.28196829739436963,
                -6.4855251203039338e-002,
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.28196829739436963,
                6.4855251203039338e-002,
                -0.28196829739436963,
                -6.4855251203039338e-002,
                -0.28196829739436963,
                -6.4855251203039338e-002,
                0.28196829739436963,
                6.4855251203039338e-002,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                -0.28196829739436963,
                -6.4855251203039338e-002,
                -0.28196829739436963,
                -6.4855251203039338e-002,
                0.28196829739436963,
                6.4855251203039338e-002,
                0.28196829739436963,
                6.4855251203039338e-002,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.26132497245657471,
                0.12150281739592229,
                0.26132497245657471,
                0.12150281739592229,
                -0.26132497245657471,
                -0.12150281739592229,
                -0.26132497245657471,
                -0.12150281739592229,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000,
                0.0000000000000000,
                -0.26132497245657471,
                -0.12150281739592229,
                0.26132497245657471,
                0.12150281739592229,
                0.26132497245657471,
                0.12150281739592229,
                -0.26132497245657471,
                -0.12150281739592229,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                0.0000000000000000,
                1.0000000000000000,
                -0.26132497245657471,
                -0.12150281739592229,
                0.26132497245657471,
                0.12150281739592229,
                -0.26132497245657471,
                -0.12150281739592229,
                0.26132497245657471,
                0.12150281739592229,
                0.42641210131950702,
                0.28196829739436963,
                0.28196829739436963,
                -0.28196829739436963,
                0.0000000000000000,
                0.0000000000000000,
                0.26132497245657471,
                -0.26132497245657471,
                -0.26132497245657471,
                1.0000000000000000,
                0.0000000000000000,
                6.6373990020049747e-002,
                -9.5770489410828816e-002,
                6.6373990020049747e-002,
                -9.5770489410828816e-002,
                6.6373990020049747e-002,
                -9.5770489410828816e-002,
                -0.14556746567207857,
                6.4855251203039338e-002,
                6.4855251203039338e-002,
                -6.4855251203039338e-002,
                0.0000000000000000,
                0.0000000000000000,
                0.12150281739592229,
                -0.12150281739592229,
                -0.12150281739592229,
                0.0000000000000000,
                1.0000000000000000,
                -9.5770489410828885e-002,
                2.7099986420623501e-002,
                -9.5770489410828885e-002,
                2.7099986420623501e-002,
                -9.5770489410828885e-002,
                2.7099986420623501e-002,
                0.42641210131950702,
                -0.28196829739436963,
                -0.28196829739436963,
                -0.28196829739436963,
                0.0000000000000000,
                0.0000000000000000,
                0.26132497245657471,
                0.26132497245657471,
                0.26132497245657471,
                6.6373990020049747e-002,
                -9.5770489410828885e-002,
                1.0000000000000000,
                0.0000000000000000,
                6.6373990020049747e-002,
                -9.5770489410828816e-002,
                6.6373990020049747e-002,
                -9.5770489410828816e-002,
                -0.14556746567207857,
                -6.4855251203039338e-002,
                -6.4855251203039338e-002,
                -6.4855251203039338e-002,
                0.0000000000000000,
                0.0000000000000000,
                0.12150281739592229,
                0.12150281739592229,
                0.12150281739592229,
                -9.5770489410828816e-002,
                2.7099986420623501e-002,
                0.0000000000000000,
                1.0000000000000000,
                -9.5770489410828885e-002,
                2.7099986420623501e-002,
                -9.5770489410828885e-002,
                2.7099986420623501e-002,
                0.42641210131950702,
                0.28196829739436963,
                -0.28196829739436963,
                0.28196829739436963,
                0.0000000000000000,
                0.0000000000000000,
                -0.26132497245657471,
                0.26132497245657471,
                -0.26132497245657471,
                6.6373990020049747e-002,
                -9.5770489410828885e-002,
                6.6373990020049747e-002,
                -9.5770489410828885e-002,
                1.0000000000000000,
                0.0000000000000000,
                6.6373990020049747e-002,
                -9.5770489410828816e-002,
                -0.14556746567207857,
                6.4855251203039338e-002,
                -6.4855251203039338e-002,
                6.4855251203039338e-002,
                0.0000000000000000,
                0.0000000000000000,
                -0.12150281739592229,
                0.12150281739592229,
                -0.12150281739592229,
                -9.5770489410828816e-002,
                2.7099986420623501e-002,
                -9.5770489410828816e-002,
                2.7099986420623501e-002,
                0.0000000000000000,
                1.0000000000000000,
                -9.5770489410828885e-002,
                2.7099986420623501e-002,
                0.42641210131950702,
                -0.28196829739436963,
                0.28196829739436963,
                0.28196829739436963,
                0.0000000000000000,
                0.0000000000000000,
                -0.26132497245657471,
                -0.26132497245657471,
                0.26132497245657471,
                6.6373990020049747e-002,
                -9.5770489410828885e-002,
                6.6373990020049747e-002,
                -9.5770489410828885e-002,
                6.6373990020049747e-002,
                -9.5770489410828885e-002,
                1.0000000000000000,
                0.0000000000000000,
                -0.14556746567207857,
                -6.4855251203039338e-002,
                6.4855251203039338e-002,
                6.4855251203039338e-002,
                0.0000000000000000,
                0.0000000000000000,
                -0.12150281739592229,
                -0.12150281739592229,
                0.12150281739592229,
                -9.5770489410828816e-002,
                2.7099986420623501e-002,
                -9.5770489410828816e-002,
                2.7099986420623501e-002,
                -9.5770489410828816e-002,
                2.7099986420623501e-002,
                0.0000000000000000,
                1.0000000000000000,
            ]
        ).reshape((17, 17)),
        "wiberg": torch.tensor(
            [
                [
                    0.0000000000000000,
                    0.93865054674698889,
                    0.93865054674699089,
                    0.93865054674699011,
                    0.93865054674699089,
                ],
                [
                    0.93865054674698889,
                    0.0000000000000000,
                    1.7033959988060572e-002,
                    1.7033959988060138e-002,
                    1.7033959988060534e-002,
                ],
                [
                    0.93865054674699089,
                    1.7033959988060572e-002,
                    0.0000000000000000,
                    1.7033959988060347e-002,
                    1.7033959988060482e-002,
                ],
                [
                    0.93865054674699011,
                    1.7033959988060138e-002,
                    1.7033959988060347e-002,
                    0.0000000000000000,
                    1.7033959988060454e-002,
                ],
                [
                    0.93865054674699089,
                    1.7033959988060534e-002,
                    1.7033959988060482e-002,
                    1.7033959988060454e-002,
                    0.0000000000000000,
                ],
            ]
        ),
    },
}
