from .config import settings


def set_verbosity(verbosity_level: int) -> None:
    """Determines verbosity level.
    Verbosity of 0 means no output at all, except for erros.
    Verbosity of 1 means basic output containing information of each step start and end
    Verbosity of 2 means complete output, containing every information and parameter - Great for logs"""
    if 0 <= verbosity_level <= 2:
        settings.verbosity = verbosity_level
    else:
        print(f"Verbosity level must be 0, 1 or 2. Got {verbosity_level}.")
        print(f"Verbosity level is {settings.verbosity}.")


def print_low(input_string) -> None:
    if 1 <= settings.verbosity <= 2:
        print(input_string)


def print_high(input_string) -> None:
    if settings.verbosity == 2:
        print(input_string)


def _check_kwargs(kwargs: dict, arg: str, default, type_to_check: type = None, optional: bool = True) -> float:
    if type_to_check is not None:
        try:
            default = type_to_check(default)
        except ValueError as e:
            print(f"Provided default value: {default} could not be converted to {type_to_check}")
            raise e
    value = default
    try:
        if not optional:
            if arg not in kwargs.keys():
                raise ValueError(f"Non-optional argument {arg} not provided.")
        if arg not in kwargs.keys():
            print_high(f"Optional argument {arg} not provided.")
        else:
            if type_to_check is not None:
                try:
                    kwargs[arg] = type_to_check(kwargs[arg])
                except ValueError as e:
                    print_low(f"Parameter {arg} could not be converted to {str(type_to_check)}.")
                    raise e
            value = kwargs[arg]
            print_high(f"Using alpha={value} for quantile scoring.")
    except ValueError:
        print(f"Could not use provided {arg}, using standard value: {default}")
    return value

# classificação x regressão
# FILTRAGEM POR SIMILARIDADE NA BUSCA DO DATASET
# PDF
# EXPLAIN
# implementar em R
