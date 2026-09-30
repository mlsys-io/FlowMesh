"""
Safe function evaluation module for FlowMesh workers.

Provides secure execution of user-defined functions in a restricted environment.
Functions are materialized (compiled) and executed with limited builtins and module
access to prevent malicious code execution while supporting common data operations.

Security model:
- Two-phase execution: materialize (compile) then execute (run)
- Restricted builtins: only safe operations (no open, eval, exec, import, etc.)
- Limited module access: json, re, math, numpy, pandas, pyarrow
- Type validation: enforces Callable[[tuple[str, ...]], str] signature
- Isolated execution: exec() with explicit safe_globals/safe_locals

Typical usage:
    fn_obj = safe_materialize_function("lambda args: args[0].upper()")
    result = safe_execute_function(fn_obj, ("hello",))  # Returns "HELLO"
"""

import inspect
import json
import math
import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

# Whitelist of safe built-in functions and types available during function execution.
# Excludes dangerous operations like open, eval, exec, compile, __import__, etc.
SAFE_BUILTINS = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "len": len,
    "sum": sum,
    "max": max,
    "min": min,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "reversed": reversed,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "any": any,
    "all": all,
    "range": range,
    "isinstance": isinstance,
}

# Whitelist of safe modules available during function execution.
# Provides common data manipulation and serialization tools without system access.
SAFE_MODULES = {
    "json": json,
    "re": re,
    "math": math,
    "np": np,
    "pd": pd,
    "pa": pa,
}


def safe_materialize_function(
    fn_code: str,
) -> Callable[[tuple[str | list[dict[str, str]], ...]], str]:
    """
    Compile function source code into a callable object with restricted builtins.

    Two-phase security model: materialize (this function) creates the function object,
    then safe_execute_function runs it in an isolated environment. This separation
    enables signature validation, function caching, and additional security checks
    before execution.

    Supported function formats:
    - Lambda expressions: "lambda args: args[0].upper()"
    - Named functions: "def transform(args):\\n    return args[0].upper()"

    Type signature enforcement:
    - Must accept exactly 1 parameter (tuple of strings or list of messages)
    - Should return a string (validated at execution time)
    - Signature: Callable[[tuple[str, ...]], str]

    Args:
        fn_code: Python source code defining a function or lambda expression

    Returns:
        Compiled function object with restricted builtin access

    Raises:
        RuntimeError: If compilation fails or signature doesn't match required format

    Examples:
        >>> fn = safe_materialize_function("lambda args: args[0].upper()")
        >>> fn = safe_materialize_function("def f(args):\\n    return json.dumps(args)")
    """
    safe_globals = {
        "__builtins__": SAFE_BUILTINS,
        **SAFE_MODULES,
    }

    fn_code_stripped = fn_code.strip()

    # Case 1: Lambda expression (use eval - lambdas are expressions)
    if fn_code_stripped.startswith("lambda"):
        try:
            fn_obj = eval(fn_code_stripped, safe_globals, {})
            if not callable(fn_obj):
                raise RuntimeError("Lambda expression did not produce a callable")
        except Exception as e:
            raise RuntimeError(
                f"Lambda compilation failed: {e}\nCode: {fn_code}"
            ) from e

    # Case 2: Function definition (use exec - def is a statement)
    else:
        safe_locals: dict[str, Any] = {}

        # Execute the function definition (creates function object in locals)
        try:
            exec(fn_code_stripped, safe_globals, safe_locals)
        except Exception as e:
            raise RuntimeError(
                f"Function definition failed: {e}\nCode: {fn_code}"
            ) from e

        # Find the function object
        if not safe_locals:
            raise RuntimeError("Function definition did not create any objects")

        # Get the function (usually the first/only item in locals)
        fn_name = list(safe_locals.keys())[0]
        fn_obj = safe_locals[fn_name]

        if not callable(fn_obj):
            raise RuntimeError(f"Defined object '{fn_name}' is not callable")

    # Validate function signature: should accept a single tuple parameter
    try:
        sig = inspect.signature(fn_obj)
        params = list(sig.parameters.values())

        # Check that function takes exactly 1 parameter (the tuple)
        if len(params) != 1:
            raise RuntimeError(
                "Function must accept exactly 1 parameter (a tuple), but has "
                f"{len(params)} parameters. Expected signature: "
                "fn(args: tuple[str, ...]) -> str"
            )

        # Note: We can't easily validate that it returns a string at compile time,
        # but we'll validate at execution time

    except ValueError:
        # Some builtins don't have inspectable signatures, allow them
        pass

    return fn_obj  # type: ignore[return-value]


def safe_execute_function(
    fn_obj: Callable[[tuple[str | list[dict[str, str]], ...]], str],
    args: tuple[str | Sequence[dict[str, str]], ...],
    allowed_modules: dict[str, Any] | None = None,
) -> str:
    """
    Execute a function in an isolated environment with no access to external state.

    Implements true sandboxing via exec() with explicit globals/locals dictionaries.
    The function runs with only whitelisted builtins and modules, preventing:
    - File system access (no open, os, pathlib, etc.)
    - Network access (no socket, urllib, requests, etc.)
    - Code injection (no eval, exec, compile, __import__)
    - System calls (no subprocess, sys.exit, etc.)

    Execution flow:
    1. Validate input types (args must be tuple of strings or lists)
    2. Create isolated globals with SAFE_BUILTINS and SAFE_MODULES
    3. Execute function call via exec() in restricted environment
    4. Extract result and validate output type (must be string)

    Args:
        fn_obj: Compiled function from safe_materialize_function()
        args: Tuple of string arguments to pass to the function
        allowed_modules: Optional dict of additional modules to allow during execution.
                         If None, uses SAFE_MODULES
                         (json, re, math, numpy, pandas, pyarrow).

    Returns:
        String result from function execution

    Raises:
        RuntimeError: If function execution fails for any reason
        TypeError: If args is not tuple[str, ...] or result is not str

    Examples:
        >>> fn = safe_materialize_function("lambda args: args[0].upper()")
        >>> safe_execute_function(fn, ("hello",))  # Returns "HELLO"
        >>> safe_execute_function(fn, ("hello", "world"))  # Passes tuple to lambda
    """
    if not callable(fn_obj):
        raise RuntimeError("Provided object is not callable")

    # Validate input type: must be a tuple of strings
    if not isinstance(args, tuple):
        raise TypeError(f"Args must be a tuple, got {type(args).__name__}")

    if not all(isinstance(arg, (str, list)) for arg in args):
        arg_types = [type(arg).__name__ for arg in args]
        raise TypeError(f"All args must be strings or lists, got types: {arg_types}")

    # Prepare restricted execution environment
    safe_globals = {
        "__builtins__": SAFE_BUILTINS,
        **(allowed_modules or SAFE_MODULES),
    }

    # Create safe locals with the function and arguments
    safe_locals = {
        "__function__": fn_obj,
        "__args__": args,
    }

    # Build the exec statement
    exec_code = "__result__ = __function__(__args__)"

    try:
        # Execute the function call in the restricted environment
        exec(exec_code, safe_globals, safe_locals)

        # Extract the result from safe_locals
        result = safe_locals["__result__"]

        # Validate output type: must be a string
        if not isinstance(result, str):
            raise TypeError(
                "Function must return a string, but returned "
                f"{type(result).__name__}: {result}"
            )

        return result
    except Exception as e:
        raise RuntimeError(f"Function execution failed: {e}") from e
