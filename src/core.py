from .interpreter import *


class SatisfiabilityMessages:
    UNSATISFIABLE = "Unsatisfiable"
    SATISFIABLE = "Satisfiable"
    UNKNOWN = "Cannot determine satisfiability"


SatisfiabilityMapping: list[str] = [
    SatisfiabilityMessages.UNSATISFIABLE,
    SatisfiabilityMessages.SATISFIABLE,
    SatisfiabilityMessages.UNKNOWN,
]


class ValidityMessages:
    VALID = "Valid"
    NOT_VALID = "Not Valid"
    UNKNOWN = "Cannot determine validity"


validity_mapping: list[str] = [
    ValidityMessages.VALID,
    ValidityMessages.NOT_VALID,
    ValidityMessages.UNKNOWN,
]


def parse(fmla: str) -> str:
    return parse_formula(fmla)


def lhs(fmla: str) -> str:
    return get_lhs_from_connective(fmla)


def con(fmla: str) -> str:
    return get_connective(fmla)


def rhs(fmla: str) -> str:
    return get_rhs_from_connective(fmla)


def theory(fmla: str) -> ParsedNode:
    return initialize_theory(fmla)


def sat(fmla: str) -> str:
    tableau = theory(fmla)
    return SatisfiabilityMapping[is_satisfiable([tableau])]


def valid(fmla: str) -> str:
    tableau = theory(fmla)
    sat_val = is_satisfiable([ParsedNegation(tableau)])
    return validity_mapping[sat_val]
