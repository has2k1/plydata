# Credit: https://github.com/pydata/patsy

import __future__
import sys
import numbers
import inspect
from functools import lru_cache


__all__ = ["EvalEnvironment"]


def _all_future_flags():
    flags = 0
    for feature_name in __future__.all_feature_names:
        feature = getattr(__future__, feature_name)
        if feature.getMandatoryRelease() > sys.version_info:
            flags |= feature.compiler_flag
    return flags


_ALL_FUTURE_FLAGS = _all_future_flags()


@lru_cache(maxsize=256)
def _compile(source, filename, mode, flags=0, dont_inherit=False, optimize=-1):
    """
    Cached compile
    """
    return compile(source, filename, mode, flags, dont_inherit)


# This is just a minimal dict-like object that does lookup in a 'stack' of
# dicts -- first it checks the first, then the second, etc. Assignments go
# into an internal, zeroth dict.
class VarLookupDict(object):
    def __init__(self, dicts):
        self._dicts = [{}] + list(dicts)

    def __getitem__(self, key):
        for d in self._dicts:
            try:
                return d[key]
            except KeyError:
                pass
        raise KeyError(key)

    def __setitem__(self, key, value):
        self._dicts[0][key] = value

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self._dicts)

    def __getstate__(*args, **kwargs):
        raise NotImplementedError("Sorry, pickling not supported")

    def copy(self):
        return self


class EvalEnvironment(object):
    """Represents a Python execution environment.
    Encapsulates a namespace for variable lookup and set of __future__
    flags."""
    def __init__(self, namespaces, flags=0):
        assert not flags & ~_ALL_FUTURE_FLAGS
        self._namespaces = list(namespaces)
        self.flags = flags

    @property
    def namespace(self):
        """A dict-like object that can be used to look up variables accessible
        from the encapsulated environment."""
        return VarLookupDict(self._namespaces)

    def with_outer_namespace(self, outer_namespace):
        """Return a new EvalEnvironment with an extra namespace added.
        This namespace will be used only for variables that are not found in
        any existing namespace, i.e., it is "outside" them all."""
        return self.__class__(self._namespaces + [outer_namespace],
                              self.flags)

    def eval(self, expr, source_name="<string>", inner_namespace={}):
        """Evaluate some Python code in the encapsulated environment.
        :arg expr: A string containing a Python expression.
        :arg source_name: A name for this string, for use in tracebacks.
        :arg inner_namespace: A dict-like object that will be checked first
          when `expr` attempts to access any variables.
        :returns: The value of `expr`.
        """
        code = _compile(expr, source_name, 'eval', self.flags)
        return eval(code, {}, VarLookupDict([inner_namespace]
                                            + self._namespaces))

    @classmethod
    def capture(cls, eval_env=0, reference=0):
        """Capture an execution environment from the stack.
        If `eval_env` is already an :class:`EvalEnvironment`, it is returned
        unchanged. Otherwise, we walk up the stack by ``eval_env + reference``
        steps and capture that function's evaluation environment.
        For ``eval_env=0`` and ``reference=0``, the default, this captures the
        stack frame of the function that calls :meth:`capture`. If ``eval_env
        + reference`` is 1, then we capture that function's caller, etc.
        This somewhat complicated calling convention is designed to be
        convenient for functions which want to capture their caller's
        environment by default, but also allow explicit environments to be
        specified. See the second example.
        Example::
          x = 1
          this_env = EvalEnvironment.capture()
          assert this_env.namespace["x"] == 1
          def child_func():
              return EvalEnvironment.capture(1)
          this_env_from_child = child_func()
          assert this_env_from_child.namespace["x"] == 1
        Example::
          # This function can be used like:
          #   my_model(formula_like, data)
          #     -> evaluates formula_like in caller's environment
          #   my_model(formula_like, data, eval_env=1)
          #     -> evaluates formula_like in caller's caller's environment
          #   my_model(formula_like, data, eval_env=my_env)
          #     -> evaluates formula_like in environment 'my_env'
          def my_model(formula_like, data, eval_env=0):
              eval_env = EvalEnvironment.capture(eval_env, reference=1)
              return model_setup_helper(formula_like, data, eval_env)
        This is how :func:`dmatrix` works.
        .. versionadded: 0.2.0
           The ``reference`` argument.
        """
        if isinstance(eval_env, cls):
            return eval_env
        elif isinstance(eval_env, numbers.Integral):
            depth = eval_env + reference
        else:
            raise TypeError("Parameter 'eval_env' must be either an integer "
                            "or an instance of patsy.EvalEnvironment.")
        frame = inspect.currentframe()
        try:
            for i in range(depth + 1):
                if frame is None:
                    raise ValueError("call-stack is not that deep!")
                frame = frame.f_back
            return cls([frame.f_locals, frame.f_globals],
                       frame.f_code.co_flags & _ALL_FUTURE_FLAGS)
        # The try/finally is important to avoid a potential reference cycle --
        # any exception traceback will carry a reference to *our* frame, which
        # contains a reference to our local variables, which would otherwise
        # carry a reference to some parent frame, where the exception was
        # caught...:
        finally:
            del frame

    def subset(self, names):
        """Creates a new, flat EvalEnvironment that contains only
        the variables specified."""
        vld = VarLookupDict(self._namespaces)
        new_ns = dict((name, vld[name]) for name in names)
        return EvalEnvironment([new_ns], self.flags)

    def _namespace_ids(self):
        return [id(n) for n in self._namespaces]

    def __eq__(self, other):
        return (isinstance(other, EvalEnvironment)
                and self.flags == other.flags
                and self._namespace_ids() == other._namespace_ids())

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((EvalEnvironment,
                     self.flags,
                     tuple(self._namespace_ids())))

    def __getstate__(*args, **kwargs):
        raise NotImplementedError("Sorry, pickling not supported")
