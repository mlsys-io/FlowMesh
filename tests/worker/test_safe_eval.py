"""safe_eval tests: the string-result prompt path and the list-result Lambda path."""

import pytest

from worker.executors.utils.safe_eval import (
    safe_execute_function,
    safe_materialize_function,
)


def _run(fn_code: str, args: tuple, *, expect_list: bool = False):
    fn_obj = safe_materialize_function(fn_code)
    return safe_execute_function(fn_obj, args, expect_list=expect_list)


class TestStringResult:
    def test_string_result_is_returned(self) -> None:
        assert _run("lambda args: args[0].upper()", ("hello",)) == "HELLO"

    def test_non_string_result_raises(self) -> None:
        with pytest.raises(RuntimeError, match="must return a string"):
            _run("lambda args: [1, 2]", ([1],))


class TestListResult:
    def test_list_of_json_is_returned(self) -> None:
        assert _run(
            "lambda args: [x * 2 for x in args[0]]", ([1, 2, 3],), expect_list=True
        ) == [2, 4, 6]

    def test_groups_are_allowed_as_elements(self) -> None:
        assert _run("lambda args: [[1, 2], [3, 4, 5]]", ([],), expect_list=True) == [
            [1, 2],
            [3, 4, 5],
        ]

    def test_non_list_result_raises(self) -> None:
        with pytest.raises(RuntimeError, match="must return a list"):
            _run("lambda args: 'nope'", ([],), expect_list=True)

    def test_non_json_element_raises(self) -> None:
        with pytest.raises(RuntimeError, match="list of JSON"):
            _run("lambda args: [set([1])]", ([],), expect_list=True)

    def test_nested_non_json_element_raises(self) -> None:
        with pytest.raises(RuntimeError, match="list of JSON"):
            _run("lambda args: [{'k': set([1])}]", ([],), expect_list=True)

    def test_dict_argument_reaches_function(self) -> None:
        assert _run("lambda args: [args[0]['a']]", ({"a": 1},), expect_list=True) == [1]

    def test_scalar_argument_reaches_function(self) -> None:
        assert _run("lambda args: [args[0]]", (5,), expect_list=True) == [5]

    def test_string_mode_rejects_dict_argument(self) -> None:
        with pytest.raises(TypeError, match="strings or lists"):
            _run("lambda args: args[0]['a']", ({"a": 1},))


class TestMaterializeShape:
    def test_single_def_works_in_string_mode(self) -> None:
        assert _run("def f(args):\n    return args[0].upper()", ("hi",)) == "HI"

    def test_single_def_works_in_function_mode(self) -> None:
        assert _run(
            "def f(args):\n    return [x * 2 for x in args[0]]",
            ([1, 2],),
            expect_list=True,
        ) == [2, 4]

    def test_lambda_works_in_both_modes(self) -> None:
        assert _run("lambda args: args[0].upper()", ("hi",)) == "HI"
        assert _run("lambda args: [args[0]]", (1,), expect_list=True) == [1]

    def test_two_defs_raise(self) -> None:
        with pytest.raises(RuntimeError, match="single function definition"):
            _run(
                "def f(args):\n    return args[0]\ndef g(args):\n    return args[0]",
                (1,),
            )

    def test_def_plus_top_level_statement_raises(self) -> None:
        with pytest.raises(RuntimeError, match="single function definition"):
            _run("def f(args):\n    return args[0]\nx = 1", (1,))

    def test_assigned_lambda_works_in_string_mode(self) -> None:
        assert _run("lam = lambda args: args[0].lower()", ("HI",)) == "hi"

    def test_assigned_lambda_works_in_function_mode(self) -> None:
        assert _run(
            "lam = lambda args: [x * 2 for x in args[0]]",
            ([1, 2],),
            expect_list=True,
        ) == [2, 4]

    def test_assign_of_non_lambda_raises(self) -> None:
        with pytest.raises(RuntimeError, match="single function definition"):
            _run("f = 1", (1,))

    def test_syntax_error_raises_runtime_error(self) -> None:
        with pytest.raises(RuntimeError, match="Function definition failed"):
            _run("def f(args):\n    return )", (1,))


class TestValueErrorPropagation:
    def test_value_error_message_survives_sandbox(self) -> None:
        """A function raising ValueError fails closed with its message intact."""
        with pytest.raises(
            RuntimeError,
            match="kept ids not in the candidate table: \\['f6'\\]",
        ):
            _run(
                "def f(args):\n"
                "    raise ValueError(\"kept ids not in the candidate table: ['f6']\")",
                (),
            )
