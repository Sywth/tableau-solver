from src import con, lhs, rhs, sat, valid


def print_help():
    help_text = """
    Commands:
      help - print this help message
      exit - exit the program
      con <formula> - check if the formula is valid
      lhs <formula> - check if the formula is a theorem
      rhs <formula> - parse the formula
      sat <formula> - check if formula is satisfiable
      valid <formula> - check if formula is valid
    """
    print(help_text)


def print_intro():
    intro_text = """
    ***===FOL Theorem Prover===***
    Created 2023-10-01 and ported on 2024-05-01
    Type 'help' for a list of commands
    """
    print(intro_text)


command_mapping = {
    "help": print_help,
    "exit": exit,
    "con": con,
    "lhs": lhs,
    "rhs": rhs,
    "sat": sat,
    "valid": valid,
}


def repl():
    """Read, Evaluate, Print Loop (REPL) for handling user commands in the theorem prover."""
    print_intro()
    while True:
        try:
            line = input(">>> ").strip()
            if not line:
                continue
            command, *args = line.split(maxsplit=1)
            formula = args[0] if args else None
            if command in command_mapping:
                result = (
                    command_mapping[command](formula)
                    if formula
                    else command_mapping[command]()
                )
                if result is not None:
                    print(result)
            else:
                print(f"Unknown command: {command}")
        except ValueError:
            print("Invalid command syntax. Type 'help' for command usage.")
        except Exception as e:
            print(f"Unexpected Error: {e}")


if __name__ == "__main__":
    repl()
