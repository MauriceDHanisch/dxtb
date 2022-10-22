"""
Molecules for testing the solvation module.
"""

import torch

from ..molecules import merge_nested_dicts, mols

from xtbml.typing import Molecule, Tensor, TypedDict


class Refs(TypedDict):
    """Format of reference records containing GFN1-xTB and GFN2-xTB reference values."""

    energies: Tensor
    """Energy vector for each atom partial charge."""

    gradient: Tensor
    """Gradient of the energy."""

    gsolv: Tensor
    """Solvation energy."""

    charges: Tensor
    """Partial charges."""

    born: Tensor
    """Reference values for Born radii"""

    psi: Tensor
    """Reference values for psi"""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "MB16_43_01": {
        "gradient": torch.tensor(
            [
                [-0.00153255497571, +0.00297789159231, -0.00025223451667],
                [+0.04018865153193, -0.00417188368738, -0.01862835511565],
                [+0.00837368890643, +0.11240474134684, -0.06477533280849],
                [-0.02703781053424, +0.01822604984045, -0.02073055319488],
                [-0.00798136088997, -0.01197965722531, +0.01962486468256],
                [-0.06074409186840, -0.04618510603905, -0.03181952610612],
                [+0.00332407513633, +0.00899094715714, +0.02887662686408],
                [-0.05291913077235, -0.02901019901037, +0.03983597084880],
                [+0.00134278449696, -0.01168669387698, -0.02276383154094],
                [+0.07952973246574, +0.02427736669779, -0.02449656277895],
                [-0.00452421233058, +0.00195662444457, +0.08199722319841],
                [-0.00408392306417, -0.01433996297419, -0.01897890307009],
                [-0.01418318692595, -0.01913838088512, +0.01145836710930],
                [-0.02751979418099, -0.05575989186764, +0.07116948813200],
                [+0.05555135011673, +0.03116076625884, -0.04667171090841],
                [+0.01221577543765, -0.00772261759266, -0.00384555035271],
            ]
        ),
        "gsolv": torch.tensor(
            [
                +0.04128841437037,
                -0.00796504045094,
                -0.06934136258348,
                +0.01172341550515,
                -0.04708187587279,
                +0.01231754324294,
                +0.00545026229412,
                -0.04903004745543,
                -0.00028009456411,
                +0.01686767488525,
                +0.00184300201445,
                +0.00324689730305,
                +0.00753672930741,
                +0.01133325448597,
                -0.00439295951184,
                +0.00222600674975,
            ]
        ),
        "energies": torch.tensor(
            [
                -0.02200339637451,
                -0.00053846697630,
                -0.00735719826976,
                -0.00083292867427,
                -0.00634305452328,
                +0.00029193446230,
                -0.00044835889449,
                -0.01256188445466,
                -0.00245665657834,
                +0.00148111264143,
                +0.00245520681858,
                -0.00016245618639,
                +0.00153900498885,
                +0.00087715042795,
                -0.00626537040940,
                +0.00325295650828,
            ]
        ),
        "charges": torch.tensor(
            [
                +7.73347900345264e-1,
                +1.07626888948184e-1,
                -3.66999593831010e-1,
                +4.92833325937897e-2,
                -1.83332156197733e-1,
                +2.33302086605469e-1,
                +6.61837152062315e-2,
                -5.43944165050002e-1,
                -2.70264356583716e-1,
                +2.66618968841682e-1,
                +2.62725033202480e-1,
                -7.15315510172571e-2,
                -3.73300777019193e-1,
                +3.84585237785621e-2,
                -5.05851088366940e-1,
                +5.17677238544189e-1,
            ]
        ),
        "born": torch.tensor(
            [
                4.0733179853143833,
                2.5645165042916651,
                2.9767695444888163,
                2.8883311275959214,
                2.5912747601100778,
                2.6375027942550999,
                3.5614957103602465,
                2.8909095828137281,
                4.5381559228327690,
                2.4634284772030313,
                2.7270746125152177,
                3.5293393256453176,
                3.6693491914686773,
                3.6669782787601948,
                3.3780928475676446,
                4.5793201340354424,
            ]
        ),
        "psi": torch.tensor(
            [
                8.1589728518833962e-002,
                0.18725100988607701,
                0.17643483830411014,
                0.25385997093494928,
                0.15017395988593091,
                0.20497561013082585,
                0.33439911953844453,
                0.15980628348605080,
                0.29303699911297815,
                0.15921868147454629,
                0.22432943630070729,
                0.11576977401414151,
                0.11354089465871361,
                0.11322683604308532,
                0.20032842544403756,
                9.1330919858808965e-002,
            ]
        ),
    },
    "MB16_43_02": {
        "gradient": torch.tensor([]),
        "gsolv": torch.tensor([]),
        "energies": torch.tensor(
            [
                +1.47620109527236e-4,
                -7.33095645702693e-4,
                -1.70970515572092e-3,
                -1.14366193957396e-2,
                -1.10697907913040e-2,
                +1.37024255683766e-3,
                +5.72539772886035e-4,
                +2.02660059874183e-4,
                +7.99504833236006e-5,
                +1.74901876175837e-3,
                -2.33115471046013e-3,
                -7.75582667617163e-3,
                -2.41207902979325e-3,
                -9.34384898116227e-5,
                +4.41237944616590e-4,
                -3.16940603695238e-3,
            ]
        ),
        "charges": torch.tensor(
            [
                +7.38394711236234e-2,
                -1.68354976558608e-1,
                -3.47642833746823e-1,
                -7.05489267186003e-1,
                +7.73548301641266e-1,
                +2.30207581365386e-1,
                +1.02748501676354e-1,
                +9.47818107467040e-2,
                +2.44260351729187e-2,
                +2.34984927037408e-1,
                -3.17839896393030e-1,
                +6.67112994818879e-1,
                -4.78119977010488e-1,
                +6.57536027459275e-2,
                +1.08259054549882e-1,
                -3.58215329983396e-1,
            ]
        ),
        "born": torch.tensor(
            [
                2.80167922633069e0,
                4.18650148856643,
                3.97512065347479,
                2.90725165315080,
                4.19750818682302,
                2.79307889405401,
                2.65183166184261,
                2.52268285366044,
                4.38725099698861,
                2.75744187621669,
                3.80708258121315,
                3.91155747499423,
                2.78077850499691,
                2.55620671005682,
                3.80403388029348,
                3.73108898118882,
            ]
        ),
        "psi": torch.tensor(
            [
                0.23874554988810831,
                0.15231307450573561,
                0.14825308202021664,
                0.16311871321883092,
                0.12739402386758411,
                0.23715426967768719,
                0.20823825804931140,
                0.17617913034384447,
                0.10195591485559931,
                0.23036867896690746,
                0.13050892653234619,
                9.5527405890196038e-002,
                0.19511432398822573,
                0.18510761926294442,
                0.35398294635573041,
                0.10437172832421032,
            ]
        ),
    },
    "SiH4": {
        "gradient": torch.tensor([]),
        "gsolv": torch.tensor([]),
        "energies": torch.tensor([]),
        "charges": torch.tensor([]),
        "born": torch.tensor(
            [
                3.66468951140369,
                2.46212969258548,
                2.46212969258548,
                2.46212969258548,
                2.46212969258548,
            ]
        ),
        "psi": torch.tensor(
            [
                1.2700510644863855e-002,
                0.15882905767115774,
                0.15882905767115774,
                0.15882905767115774,
                0.15882905767115774,
            ]
        ),
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
