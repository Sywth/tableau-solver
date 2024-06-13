"""
i apologize for then length 
========================================================
Constants for Propositional Logic and First Order Logic
========================================================
"""

from .constants import MAX_CONSTANTS


class LANGUAGE_CONSTANTS:
    class SYMBOLS:
        SYMBOL_NEGATION = "~"
        SYMBOL_OPEN_BRACKET = "("
        SYMBOL_CLOSE_BRACKET = ")"
        SYMBOL_CONJUNCTION = "/\\"
        SYMBOL_DISJUNCTION = "\\/"
        SYMBOL_IMPLICATION = "=>"
        SYMBOL_EXISTENTIAL = "E"
        SYMBOL_UNIVERSAL = "A"

    class LETTERS_PROPOSITION:
        LETTER_p = "p"
        LETTER_q = "q"
        LETTER_r = "r"
        LETTER_s = "s"

    class LETTERS_PREDICATE:
        LETTER_P = "P"
        LETTER_Q = "Q"
        LETTER_R = "R"
        LETTER_S = "S"

    class LETTERS_VARIABLE:
        LETTER_x = "x"
        LETTER_y = "y"
        LETTER_z = "z"
        LETTER_w = "w"

    PREDICATE_VARIABLE_SEPERATOR = ","


SET_LANGUAGE_SYMBOLS: set[str] = {
    LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_NEGATION,
    LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_OPEN_BRACKET,
    LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_CLOSE_BRACKET,
    LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_CONJUNCTION,
    LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_DISJUNCTION,
    LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_IMPLICATION,
    LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_EXISTENTIAL,
    LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_UNIVERSAL,
}

SET_LANGUAGE_PROPOSITION_LETTERS: set[str] = {
    LANGUAGE_CONSTANTS.LETTERS_PROPOSITION.LETTER_p,
    LANGUAGE_CONSTANTS.LETTERS_PROPOSITION.LETTER_q,
    LANGUAGE_CONSTANTS.LETTERS_PROPOSITION.LETTER_r,
    LANGUAGE_CONSTANTS.LETTERS_PROPOSITION.LETTER_s,
}

SET_LANGUAGE_PREDICATE_LETTERS: set[str] = {
    LANGUAGE_CONSTANTS.LETTERS_PREDICATE.LETTER_P,
    LANGUAGE_CONSTANTS.LETTERS_PREDICATE.LETTER_Q,
    LANGUAGE_CONSTANTS.LETTERS_PREDICATE.LETTER_R,
    LANGUAGE_CONSTANTS.LETTERS_PREDICATE.LETTER_S,
}

SET_LANGUAGE_VARIABLE_LETTERS: set[str] = {
    LANGUAGE_CONSTANTS.LETTERS_VARIABLE.LETTER_x,
    LANGUAGE_CONSTANTS.LETTERS_VARIABLE.LETTER_y,
    LANGUAGE_CONSTANTS.LETTERS_VARIABLE.LETTER_z,
    LANGUAGE_CONSTANTS.LETTERS_VARIABLE.LETTER_w,
}

SET_OF_ALL_LANGUAGE_TOKENS: set[str] = (
    SET_LANGUAGE_SYMBOLS
    | SET_LANGUAGE_PROPOSITION_LETTERS
    | SET_LANGUAGE_PREDICATE_LETTERS
    | SET_LANGUAGE_VARIABLE_LETTERS
)
SET_OF_ALL_BINARY_CONNECTIVES: set[str] = {
    LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_CONJUNCTION,
    LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_DISJUNCTION,
    LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_IMPLICATION,
}
SET_OF_ALL_QUANTIFIERS: set[str] = {
    LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_EXISTENTIAL,
    LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_UNIVERSAL,
}

"""
========================================================
Classes for parsing
========================================================
"""


class ParsingError(Exception): ...


class ParsedNode:  # Abstract Class
    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        raise NotImplementedError("ParsedNode is an abstract class")


class ParsedLiteral(ParsedNode):  # Abstract Class
    def __init__(self, symbol: str) -> None:
        super().__init__()
        self.symbol: str = symbol

    def __str__(self) -> str:
        return str(self.symbol)

    def __eq__(self, other: ParsedNode) -> bool:
        if type(self) == type(other):
            return self.symbol == other.symbol
        return False

    def __hash__(self) -> int:
        return hash(str(self))


class ParsedPropositionLetter(ParsedLiteral):
    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)


class ParsedVariable(ParsedLiteral):
    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)


class Constant(ParsedLiteral):
    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)

    def copy(self) -> "Constant":
        return Constant(self.symbol)


class UsedAllConstants(Exception): ...


class ConstantGenerator:
    MAX_CONSTANTS = 10

    def __init__(self) -> None:
        self.constants: list[Constant] = []
        self.id_to_next: dict[int, int] = {}

    def get_new(self) -> Constant:
        new_constant = Constant(f"c{len(self.constants)}")
        self.constants.append(new_constant)
        if len(self.constants) > self.MAX_CONSTANTS:
            raise ConstantLimitReached(f"Used all constants for {self.constants}")
        return new_constant

    def copy(self) -> "ConstantGenerator":
        """returns a deepcopy of self

        Returns:
            ConstantGenerator: With the same constants (usefull for branching)
        """
        new_generator = ConstantGenerator()
        new_generator.constants = [c.copy() for c in self.constants]
        new_generator.id_to_next = self.id_to_next.copy()
        return new_generator

    # def get_bound(self, id) -> Constant:
    #     """Get a new bound constant for a universal quantifier (determiend by its id)

    #     Args:
    #         id (_type_): UniversalQuantifier::get_id(branch_id)

    #     Raises:
    #         UsedAllConstants: If all constants are used up

    #     Returns:
    #         Constant: A used consant not previously generated by this specific quantifier
    #     """
    #     if id not in self.id_to_next:
    #         self.id_to_next[id] = 0

    #     idx = self.id_to_next[id]
    #     if idx >= len(self.constants):
    #         raise UsedAllConstants(f"Used all constants for id {id}")
    #     self.id_to_next[id] += 1
    #     return self.constants[idx]


class ParsedUnaryNode(ParsedNode):  # Abstract Class
    def __init__(self, child: ParsedNode) -> None:
        super().__init__()
        self.child: ParsedNode | "BinaryPredicate" | "ParsedUnboundBinaryPredicate" = (
            child
        )

    def __eq__(self, other: ParsedNode) -> bool:
        # check if they are exact same type and have the same child
        if type(self) == type(other):
            return self.child == other.child
        return False

    def __hash__(self) -> int:
        return hash(str(self))


class ParsedBinaryNode(ParsedNode):  # Abstract Class
    def __init__(self, left: ParsedNode, right: ParsedNode) -> None:
        super().__init__()
        self.left: ParsedNode = left
        self.right: ParsedNode = right

    def __eq__(self, other: ParsedNode) -> bool:
        if type(self) == type(other):
            return self.left == other.left and self.right == other.right
        return False

    def __hash__(self) -> int:
        return hash(str(self))


class ParsedNegation(ParsedUnaryNode):
    def __init__(
        self,
        child: ParsedNode,
    ) -> None:
        super().__init__(child)

    def __str__(self) -> str:
        return LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_NEGATION + str(self.child)


class ParsedConjunction(ParsedBinaryNode):
    def __init__(self, left: ParsedNode, right: ParsedNode) -> None:
        super().__init__(left, right)

    def __str__(self) -> str:
        return (
            LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_OPEN_BRACKET
            + str(self.left)
            + LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_CONJUNCTION
            + str(self.right)
            + LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_CLOSE_BRACKET
        )


class ParsedDisjunction(ParsedBinaryNode):
    def __init__(self, left: ParsedNode, right: ParsedNode) -> None:
        super().__init__(left, right)

    def __str__(self) -> str:
        return (
            LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_OPEN_BRACKET
            + str(self.left)
            + LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_DISJUNCTION
            + str(self.right)
            + LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_CLOSE_BRACKET
        )


class ParsedImplication(ParsedBinaryNode):
    def __init__(self, left: ParsedNode, right: ParsedNode) -> None:
        super().__init__(left, right)

    def __str__(self) -> str:
        return (
            LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_OPEN_BRACKET
            + str(self.left)
            + LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_IMPLICATION
            + str(self.right)
            + LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_CLOSE_BRACKET
        )


class ParsedQuantifier(ParsedUnaryNode):  # Abstract Class
    def __init__(self, variable: ParsedVariable, child: ParsedNode) -> None:
        super().__init__(child)
        self.variable = variable


class ParsedExistentialQuantifier(ParsedQuantifier):
    def __init__(self, variable: ParsedVariable, child: ParsedNode) -> None:
        super().__init__(variable, child)

    def __str__(self) -> str:
        return (
            LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_EXISTENTIAL
            + str(self.variable)
            + LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_OPEN_BRACKET
            + str(self.child)
            + LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_CLOSE_BRACKET
        )


class ConstantLimitReached(Exception): ...


class ParsedUniversalQuantifier(ParsedQuantifier):
    MAX_CONSTANTS = 10

    def __init__(self, variable: ParsedVariable, child: ParsedNode) -> None:
        super().__init__(variable, child)
        self.__constant_index = -1

    def __str__(self) -> str:
        return (
            LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_UNIVERSAL
            + str(self.variable)
            + LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_OPEN_BRACKET
            + str(self.child)
            + LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_CLOSE_BRACKET
        )

    def get_next_consant(self, constants):
        self.__constant_index += 1
        if self.__constant_index >= self.MAX_CONSTANTS:
            raise ConstantLimitReached(f"Used all constants for {str(self)}")
        if self.__constant_index >= len(constants):
            raise UsedAllConstants(
                f"Exhauted all avaliable cosntants in {constants} for {str(self)}"
            )
        return constants[self.__constant_index]


class BinaryPredicate(ParsedLiteral):
    def __init__(
        self,
        predicate_letter: str,
        const1: Constant,
        const2: Constant,
    ) -> None:
        self.predicate_letter = predicate_letter
        self.const1 = const1
        self.const2 = const2

    def __str__(self) -> str:
        return self.predicate_letter + str(self.const1) + str(self.const2)

    def __eq__(self, other: "BinaryPredicate") -> bool:
        if type(self) == type(other):
            return (
                self.predicate_letter == other.predicate_letter
                and self.const1 == other.const1
                and self.const2 == other.const2
            )
        return False


class ParsedUnboundBinaryPredicate(ParsedNode):
    def __init__(
        self,
        predicate_letter: str,
        parameter1: ParsedVariable,
        parameter2: ParsedVariable,
    ) -> None:
        self.predicate_letter = predicate_letter
        self.parameter1 = parameter1
        self.parameter2 = parameter2

        self.assignments: map[ParsedVariable, Constant] = {}

    def get_bound_predicate(self) -> BinaryPredicate:
        try:
            const1 = self.assignments[self.parameter1]
            const2 = self.assignments[self.parameter2]
        except:
            raise Exception("Not all variables have been assigned a constant")

        return BinaryPredicate(
            self.predicate_letter,
            const1,
            const2,
        )

    def __str__(self) -> str:
        return (
            self.predicate_letter
            + LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_OPEN_BRACKET
            + str(self.parameter1)
            + LANGUAGE_CONSTANTS.PREDICATE_VARIABLE_SEPERATOR
            + str(self.parameter2)
            + LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_CLOSE_BRACKET
        )

    def __eq__(self, other: "ParsedUnboundBinaryPredicate") -> bool:
        raise Warning(
            "Cannot compare bounded predicates, did you mean to compare predicate instance?"
        )

    def __hash__(self) -> int:
        raise Warning(
            "Cannot hash bounded predicates, did you mean to hash predicate instance?"
        )


"""
========================================================
Functions for parsing
========================================================
"""


def balanced_brackets(tokens: list[str]):
    bracket_level = 0
    for c in tokens:
        if c == LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_OPEN_BRACKET:
            bracket_level += 1
        elif c == LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_CLOSE_BRACKET:
            bracket_level -= 1
        if bracket_level < 0:
            break

    if bracket_level == 0:
        return True

    # pIGNORErint(f'Unbalanced brackets\n{line}\n{" "*(len(line)-1)}^')
    return False


def validate_predicates(line: str) -> bool:
    i = -1
    while i < len(line) - 1:
        i += 1
        if line[i] in SET_LANGUAGE_PREDICATE_LETTERS:
            if line[i + 1] != LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_OPEN_BRACKET:
                return False
            if line[i + 3] != LANGUAGE_CONSTANTS.PREDICATE_VARIABLE_SEPERATOR:
                return False
            if line[i + 5] != LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_CLOSE_BRACKET:
                return False
            i += 6
    return True


def validate_binary_connectives(tokens: list[str]) -> bool:
    open_bracket_count = 0
    close_bracket_count = 0
    binary_connective_count = 0
    predicate_count = 0
    for c in tokens:
        if c in SET_OF_ALL_BINARY_CONNECTIVES:
            binary_connective_count += 1
        if c == LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_OPEN_BRACKET:
            open_bracket_count += 1
        if c == LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_CLOSE_BRACKET:
            close_bracket_count += 1
        if c in SET_LANGUAGE_PREDICATE_LETTERS:
            predicate_count += 1
            # my tokenizer removes predicate brackets so we add them here
            open_bracket_count += 1
            close_bracket_count += 1

    # validate brackets count
    if open_bracket_count != close_bracket_count:
        return False

    number_of_expected_brackets = (binary_connective_count + predicate_count) * 2
    if number_of_expected_brackets != open_bracket_count + close_bracket_count:
        return False

    return True


def is_well_formed(line: str) -> bool:
    # HOTFIX 23/11/2023, i just want it work :(
    #   :( # TODO add bracket asummption to tokenizer after

    # check all predicates are binary predicates
    if not validate_predicates(line):
        return False

    try:
        tokens = tokenizer(line)
    except ParsingError as _:
        return False

    # check brackets are in order
    if not balanced_brackets(tokens):
        return False
    # check we the correct number of brackets
    if not validate_binary_connectives(tokens):
        return False
    return True


def tokenizer(line: str) -> list[str]:
    if line == "":
        return []

    if line[0] in SET_LANGUAGE_PREDICATE_LETTERS:
        return [line[0], line[2], line[4]] + tokenizer(line[6:])
    if line[0] in (SET_OF_ALL_LANGUAGE_TOKENS - SET_LANGUAGE_PREDICATE_LETTERS):
        return [line[0]] + tokenizer(line[1:])
    if line[0:2] in SET_OF_ALL_LANGUAGE_TOKENS:
        return [line[0:2]] + tokenizer(line[2:])

    raise ParsingError(f"Unrecognized token for line\n\t{line}\n\t^")


def tokens_to_string(tokens: list[str]) -> str:
    # TODO figure out a way to parse with brackets ? this kinda hacky

    new_tokens = []
    while tokens:
        if tokens[0] in SET_LANGUAGE_PREDICATE_LETTERS:
            new_tokens.append(tokens.pop(0))
            new_tokens.append(LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_OPEN_BRACKET)
            new_tokens.append(tokens.pop(0))
            new_tokens.append(LANGUAGE_CONSTANTS.PREDICATE_VARIABLE_SEPERATOR)
            new_tokens.append(tokens.pop(0))
            new_tokens.append(LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_CLOSE_BRACKET)
        else:
            new_tokens.append(tokens.pop(0))

    return "".join(t for t in new_tokens)


def find_matching_bracket_index(tokens: list[str]) -> int:
    """
    Will return the index of closing bracket for the first open bracket encountered,
    assumes that the first token is an open bracket
    """
    bracket_level = 0
    for i in range(len(tokens)):
        c = tokens[i]
        if c == LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_OPEN_BRACKET:
            bracket_level += 1
        elif c == LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_CLOSE_BRACKET:
            bracket_level -= 1
        if bracket_level == 0:
            return i

    raise ParsingError(
        f'Unbalanced brackets\n{tokens_to_string(tokens)}\n{" "*(len(tokens)-1)}^'
    )


def find_binary_connective_index(tokens: list[str]) -> int:
    """
    Will return the index of connective corresponding to the open bracket encountered,
    assumes that the first token is an open bracket
    """
    bracket_level = 0
    for i in range(len(tokens)):
        c = tokens[i]
        if c == LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_OPEN_BRACKET:
            bracket_level += 1
        elif c == LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_CLOSE_BRACKET:
            bracket_level -= 1
        elif bracket_level == 1 and c in SET_OF_ALL_BINARY_CONNECTIVES:
            return i

    # HOTFIX 23/11/2023, i just want it work :( # TODO find proper implementation

    raise ParsingError(
        f'Unbalanced brackets\n{tokens_to_string(tokens)}\n{" "*(len(tokens)-1)}^'
    )


def parse_binary_operation(
    lhs: ParsedNode, rhs: ParsedNode, connective: str
) -> ParsedNode | None:
    if connective == LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_CONJUNCTION:
        return ParsedConjunction(lhs, rhs)
    if connective == LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_DISJUNCTION:
        return ParsedDisjunction(lhs, rhs)
    if connective == LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_IMPLICATION:
        return ParsedImplication(lhs, rhs)

    raise ParsingError(f'Unrecognized connective "{connective}"')


def parse_tokens(tokens: list[str]) -> ParsedNode:
    if tokens == []:
        raise ParsingError("Empty tokens")

    if len(tokens) == 1:
        if tokens[0] in SET_LANGUAGE_PROPOSITION_LETTERS:
            return ParsedPropositionLetter(tokens[0])

        raise ParsingError(f'Unrecognized token \n\t{" ".join(tokens)}\n\t^')

    if tokens[0] == LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_EXISTENTIAL:
        return ParsedExistentialQuantifier(
            ParsedVariable(tokens[1]), parse_tokens(tokens[2:])
        )

    if tokens[0] == LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_UNIVERSAL:
        return ParsedUniversalQuantifier(
            ParsedVariable(tokens[1]), parse_tokens(tokens[2:])
        )

    if tokens[0] == LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_NEGATION:
        return ParsedNegation(parse_tokens(tokens[1:]))

    if tokens[0] == LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_OPEN_BRACKET:
        connective_index = find_binary_connective_index(tokens)
        lhs = parse_tokens(tokens[1:connective_index])
        rhs = parse_tokens(tokens[connective_index + 1 : -1])
        return parse_binary_operation(lhs, rhs, tokens[connective_index])

    if (
        tokens[0] in SET_LANGUAGE_PROPOSITION_LETTERS
        or tokens[0] in SET_LANGUAGE_VARIABLE_LETTERS
    ):
        connective = tokens[1]
        lhs = parse_tokens([tokens[0]])
        rhs = parse_tokens(tokens[2:])
        return parse_binary_operation(lhs, rhs, connective)

    if tokens[0] in SET_LANGUAGE_PREDICATE_LETTERS:
        var1 = tokens[1]
        var2 = tokens[2]
        return ParsedUnboundBinaryPredicate(
            tokens[0], ParsedVariable(var1), ParsedVariable(var2)
        )

    raise ParsingError(f'Unrecognized token \n\t{" ".join(tokens)}\n\t^')


def parse_line(line: str) -> ParsedNode | None:
    if not is_well_formed(line):
        raise ParsingError(f"Formula {line} is not well formed.")
    return parse_tokens(tokenizer(line))


def contains_fol_quantifier(formula: str) -> bool:
    for token in tokenizer(formula):
        if token in SET_OF_ALL_QUANTIFIERS:
            return True
    return False


def parse_formula(formula: str) -> int:
    if not is_well_formed(formula):
        return 0
    try:
        root = parse_line(formula)
    except ParsingError as _:
        return 0

    is_fol = contains_fol_quantifier(formula)

    if isinstance(root, ParsedUnboundBinaryPredicate):
        return 1
    if isinstance(root, ParsedNegation):
        return 2 if is_fol else 7
    if isinstance(root, ParsedUniversalQuantifier):
        return 3
    if isinstance(root, ParsedExistentialQuantifier):
        return 4
    if (
        isinstance(root, ParsedConjunction)
        or isinstance(root, ParsedDisjunction)
        or isinstance(root, ParsedImplication)
    ):
        return 5 if is_fol else 8
    if isinstance(root, ParsedPropositionLetter):
        return 6

    return 0


def get_lhs_from_connective(formula) -> str:
    tokens = tokenizer(formula)
    connective_index = find_binary_connective_index(tokens)
    return tokens_to_string(tokens[1:connective_index])


def get_rhs_from_connective(formula) -> str:
    tokens = tokenizer(formula)
    connective_index = find_binary_connective_index(tokens)
    return tokens_to_string(tokens[connective_index + 1 : -1])


def get_connective(formula) -> str:
    tokens = tokenizer(formula)
    connective_index = find_binary_connective_index(tokens)
    return tokens[connective_index]


def initialize_theory(formula: str) -> ParsedNode:
    return parse_line(formula)


class Path:
    """Symbolic of a path down the tableaux tree"""

    def __init__(
        self,
        unevaluated_nodes: list[ParsedNode] = None,
        literals: set[ParsedNode] = None,
    ):
        self.unevaluated_formulae: list[ParsedNode | ParsedUnboundBinaryPredicate] = (
            unevaluated_nodes if unevaluated_nodes else []
        )
        self.literals: set[str] = literals if literals else set()
        self.consant_generator: ConstantGenerator = ConstantGenerator()

    def __contradicts(
        self, literal: ParsedLiteral | ParsedNegation | BinaryPredicate
    ) -> bool:
        # a negated literal contradicts if a non-negated version of already exists
        if isinstance(literal, ParsedNegation):
            if type(literal.child) == BinaryPredicate:
                return str(literal.child) in self.literals
            if isinstance(literal.child, ParsedLiteral):
                return literal.child.symbol in self.literals

        # a literal contradicts if a negated version already exists
        if type(literal) == BinaryPredicate:
            return (
                f"{LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_NEGATION}{str(literal)}"
                in self.literals
            )
        # all other literals
        if isinstance(literal, ParsedLiteral):
            return (
                f"{LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_NEGATION}{literal.symbol}"
                in self.literals
            )

        raise Exception(
            f'Unrecognized token "{literal}", cannot determine contradiction'
        )

    def try_add_literal(
        self, literal: ParsedLiteral | ParsedNegation | BinaryPredicate
    ) -> bool:
        # if it contradicts return false
        if self.__contradicts(literal):
            return False

        # if its negated add the negation symbol (usually ¬ or ~)
        if isinstance(literal, ParsedNegation):
            if type(literal.child) == BinaryPredicate:
                self.literals.add(
                    f"{LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_NEGATION}{str(literal.child)}"
                )
                return True
            if isinstance(literal.child, ParsedLiteral):
                self.literals.add(
                    f"{LANGUAGE_CONSTANTS.SYMBOLS.SYMBOL_NEGATION}{literal.child.symbol}"
                )
                return True

        if isinstance(literal, BinaryPredicate):
            self.literals.add(str(literal))
            return True
        if isinstance(literal, ParsedLiteral):
            self.literals.add(literal.symbol)
            return True

        raise Exception(f'Unrecognized token "{literal}" cannot be added to literals')

    def copy(self) -> "Path":
        """Returns a shallow copy of self"""

        new_path = Path()
        new_path.unevaluated_formulae = self.unevaluated_formulae.copy()
        new_path.literals = self.literals.copy()
        new_path.consant_generator = self.consant_generator.copy()
        return new_path

    def __str__(self) -> str:
        litearls = ", ".join(str(l) for l in self.literals)
        unevaluated_formulae = ", ".join(str(f) for f in self.unevaluated_formulae)
        return f"Path(\n\tliterals={litearls},\n\tunevaluated_formulae={unevaluated_formulae},\n\tuniversal_quantifier_state={self.universal_quantifier_state},\n\tvariable_mapping={self.variable_mapping}\n\t)"


def map_all_children(variable: ParsedVariable, constant: Constant, formula: ParsedNode):
    if isinstance(formula, ParsedUnboundBinaryPredicate):
        if formula.parameter1 == variable:
            formula.assignments[formula.parameter1] = constant
        if formula.parameter2 == variable:
            formula.assignments[formula.parameter2] = constant
        return
    if isinstance(formula, ParsedExistentialQuantifier) or isinstance(
        formula, ParsedUniversalQuantifier
    ):
        # if the variable is already bound, dont bind it again
        if formula.variable == variable:
            return
    if isinstance(formula, ParsedUnaryNode):
        map_all_children(variable, constant, formula.child)
        return
    if isinstance(formula, ParsedBinaryNode):
        map_all_children(variable, constant, formula.left)
        map_all_children(variable, constant, formula.right)
        return


def is_satisfiable(tableau: list[ParsedNode]) -> int:
    paths: list[Path] = [
        Path(
            unevaluated_nodes=tableau,
            literals=set(),
        )
    ]

    while paths:
        this_path = paths.pop(0)

        branched_off: bool = False
        this_path_modified: bool = False
        for i in range(len(this_path.unevaluated_formulae) - 1, -1, -1):
            # cast enumerate to a list so we can use reverse on it to avoid index errors as we mutate the list
            formula = this_path.unevaluated_formulae.pop(i)

            # === START FORMULA SWITCH STATEMENT === #

            # Any fomrulae of type ~f
            if isinstance(formula, ParsedNegation):
                """Handle all negated formuale"""

                # Negated Litearl or Unbound Predicate : ~p , ~Pxy
                if (
                    isinstance(formula.child, ParsedLiteral)
                    or type(formula.child) == ParsedUnboundBinaryPredicate
                ):
                    # Apply bindign to unbound predicate, e.g. Pxx |-> Pcc
                    if type(formula.child) == ParsedUnboundBinaryPredicate:
                        formula.child = formula.child.get_bound_predicate()

                    # if contradiction found, stop evaluating this branch (break out of for loop)
                    if this_path.try_add_literal(formula) == False:
                        break
                    # No contradiction and we have no formulae left to evaluate on this path <-> input is satisfiable. Hence return SATISFIABLE
                    if this_path.unevaluated_formulae == []:
                        return 1
                    continue

                # Negated Existential Quantifier : ~Exf(x) = Ax~f(x)
                if isinstance(formula.child, ParsedExistentialQuantifier):
                    this_path.unevaluated_formulae.append(
                        ParsedUniversalQuantifier(
                            formula.child.variable, ParsedNegation(formula.child.child)
                        )
                    )

                    this_path_modified = True
                    continue

                # Negated Universal Quantifier : ~Axf(x) = Ex~f(x)
                if isinstance(formula.child, ParsedUniversalQuantifier):
                    this_path.unevaluated_formulae.append(
                        ParsedExistentialQuantifier(
                            formula.child.variable, ParsedNegation(formula.child.child)
                        )
                    )

                    this_path_modified = True
                    continue

                # Double Negation : ~~f === f
                if isinstance(formula.child, ParsedNegation):
                    this_path.unevaluated_formulae.append(formula.child.child)

                    this_path_modified = True
                    continue

                # Negated Disjunction : ~(f \/ g) === ~f /\ ~g
                if isinstance(formula.child, ParsedDisjunction):
                    this_path.unevaluated_formulae.append(
                        ParsedNegation(formula.child.left)
                    )
                    this_path.unevaluated_formulae.append(
                        ParsedNegation(formula.child.right)
                    )

                    this_path_modified = True
                    continue

                # Negated Implication : ~(f -> g) === f /\ ~g
                if isinstance(formula.child, ParsedImplication):
                    this_path.unevaluated_formulae.append(formula.child.left)
                    this_path.unevaluated_formulae.append(
                        ParsedNegation(formula.child.right)
                    )

                    this_path_modified = True
                    continue

                branched_off = True
                # Negated Conjunction : ~(f /\ g) === ~f \/ ~g
                if isinstance(formula.child, ParsedConjunction):
                    lhs: Path = this_path.copy()
                    rhs: Path = this_path.copy()

                    lhs.unevaluated_formulae.append(ParsedNegation(formula.child.left))
                    rhs.unevaluated_formulae.append(ParsedNegation(formula.child.right))

                    paths.append(lhs)
                    paths.append(rhs)
                    continue

                raise Exception(f'Unrecognized neagtion "{formula}"')

            """Handle all non negated formuale"""

            # Literal or Unbound Predicate : p, Pxy
            if (
                isinstance(formula, ParsedLiteral)
                or type(formula) == ParsedUnboundBinaryPredicate
            ):
                # Apply bindign to unbound predicate, e.g. Pxx |-> Pcc
                if type(formula) == ParsedUnboundBinaryPredicate:
                    formula = formula.get_bound_predicate()

                # if contradiction found, stop evaluating this branch (break out of for loop hence check next branch)
                if this_path.try_add_literal(formula) == False:
                    break
                # No contradiction and we have no formulae left to evaluate on this path <-> input is satisfiable. Hence return SATISFIABLE
                if this_path.unevaluated_formulae == [] and not branched_off:
                    return 1
                continue

            # Apply Existential Quantifier : Exf(x) = f(c) [where c not is a new constant]
            if type(formula) == ParsedExistentialQuantifier:
                try:
                    new_const = this_path.consant_generator.get_new()
                except ConstantLimitReached as e:
                    return 2

                map_all_children(formula.variable, new_const, formula.child)

                this_path.unevaluated_formulae.append(formula.child)
                this_path_modified = True
                continue

            # Apply Universal Quantifier : Axf(x) = f(c) /\ Axf(x) [where c not is a new constant]
            if isinstance(formula, ParsedUniversalQuantifier):
                try:
                    const = formula.get_next_consant(
                        this_path.consant_generator.constants
                    )
                    map_all_children(formula.variable, const, formula.child)
                except UsedAllConstants as e:
                    if this_path.unevaluated_formulae == []:
                        if (
                            this_path_modified == True
                        ):  # we dont check branched_off because we only care about this path
                            this_path.unevaluated_formulae.insert(0, formula)
                            continue
                        # no formula left to evaluate, this path is not being put back in hence, this is an open path so its satisfiable
                        return 1
                except ConstantLimitReached as e:
                    return 2

                this_path.unevaluated_formulae.append(formula.child)
                this_path.unevaluated_formulae.insert(
                    0, formula
                )  # prepend the quantifier to the front of the list (so it gets evaluated last)
                continue

            # Conjunction : f /\ g
            if isinstance(formula, ParsedConjunction):
                this_path.unevaluated_formulae.append(formula.left)
                this_path.unevaluated_formulae.append(formula.right)

                this_path_modified = True
                continue

            branched_off = True
            # Disjunction : f \/ g
            if isinstance(formula, ParsedDisjunction):
                lhs: Path = this_path.copy()
                rhs: Path = this_path.copy()

                lhs.unevaluated_formulae.append(formula.left)
                rhs.unevaluated_formulae.append(formula.right)

                paths.append(lhs)
                paths.append(rhs)
                continue

            # Implication : f -> g
            if isinstance(formula, ParsedImplication):
                lhs: Path = this_path.copy()
                rhs: Path = this_path.copy()

                lhs.unevaluated_formulae.append(ParsedNegation(formula.left))
                rhs.unevaluated_formulae.append(formula.right)

                paths.append(lhs)
                paths.append(rhs)
                continue

            # === END FORMULA SWITCH STATEMENT === #
            # if the formula doesn't match anything we'd expect
            raise Exception(f'Unrecognized formula "{formula}"')

        # This branch had more than just literals/predicates inside of it. Evaulate again in another iteration
        if (not branched_off) and this_path_modified:
            paths.append(this_path)

    # No paths left to evaluate, didn't get to end of open branch (else would have already returned) <-> input is UNSATISFIABLE
    return 0
