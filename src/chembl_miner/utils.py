from .config import settings


def set_verbosity(verbosity_level: int) -> None:
    """
    Sets the global verbosity level for the package.

    - 0: No output except for errors.
    - 1: Basic output (e.g., step start/end).
    - 2: Complete output (e.g., parameters, detailed info).

    Args:
        verbosity_level (int): The verbosity level (0, 1, or 2).
    
    Example:
        ```python
        from chembl_miner.utils import set_verbosity, print_low
        
        # Set verbosity to minimal
        set_verbosity(1)
        
        # This will now print
        print_low("Pipeline started.")
        
        # Set verbosity to off
        set_verbosity(0)
        
        # This will no longer print
        print_low("Pipeline finished.")
        ```
    """
    if 0 <= verbosity_level <= 2:
        settings.verbosity = verbosity_level
    else:
        print(f"Verbosity level must be 0, 1 or 2. Got {verbosity_level}.")
        print(f"Verbosity level is {settings.verbosity}.")


def print_low(input_string) -> None:
    """
    Prints a message if the global verbosity is 1 or 2.

    Args:
        input_string: The string or object to print.
    """

    if 1 <= settings.verbosity <= 2:
        print(input_string)


def print_high(input_string) -> None:
    """
    Prints a message only if the global verbosity is 2.

    Args:
        input_string: The string or object to print.
    """

    if settings.verbosity == 2:
        print(input_string)


def _check_kwargs(kwargs: dict, arg: str, default, type_to_check: type = None, optional: bool = True) -> float:
    """
    Internal helper to safely get and type-check a value from a kwargs dict.

    Args:
        kwargs (dict): The kwargs dictionary.
        arg (str): The key (argument name) to look for.
        default: The default value to return if `arg` is not in `kwargs`.
        type_to_check (type, optional): A type (e.g., `float`) to
            which the value should be cast. Defaults to None.
        optional (bool, optional): If False, raises a ValueError if `arg`
            is not in `kwargs`. Defaults to True.

    Returns:
        The value from `kwargs` (or the default), cast to the
            specified type.
    
    Example:
        ```python
        # This is an internal helper function.
        
        def my_function(**kwargs):
            # Get 'alpha', default to 0.5, and ensure it's a float
            alpha = _check_kwargs(
                kwargs, 'alpha', 0.5, type_to_check=float
            )
            
            # Get 'n_jobs', default to 1, ensure it's an int
            n_jobs = _check_kwargs(
                kwargs, 'n_jobs', 1, type_to_check=int
            )
            
            return alpha, n_jobs

        # Call the function
        a, n = my_function(alpha="0.25", other_arg=True)
        # a is 0.25 (as float)
        # n is 1 (default)
        ```
    """
    
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
            print_high(f"Using {arg}={value}")
    except ValueError:
        print(f"Could not use provided {arg}, using standard value: {default}")
    return value

# classificação x regressão
# FILTRAGEM POR SIMILARIDADE NA BUSCA DO DATASET
# PDF
# EXPLAIN
# implementar em R
